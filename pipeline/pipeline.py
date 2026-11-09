import os, re, json, time, random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.ensemble import RandomForestClassifier

SITES = ['site_1_oba', 'site_2_odo_otin', 'site_3_oyo_west', 'site_4_egbeda',
         'site_5_ikere_gorge', 'site_6_sepeteri', 'site_7_kishi', 'site_8_orelope']
SEASONS = {'dry': {11, 12, 1, 2, 3}, 'wet': {4, 5, 6, 9, 10}}
BINS = ['0%', '0-1%', '1-5%', '5-25%', '25-50%', '>50%']
EDGES = [(0.01, '0-1%'), (0.05, '1-5%'), (0.25, '5-25%'), (0.50, '25-50%')]
IDX = {'ndwi': 5, 'mndwi': 6, 'awei': 8}
UNET_GRID = np.round(np.concatenate([np.arange(0.05, 0.951, 0.05), [0.97, 0.99]]), 2)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CFG = {}
_INDEX = {}


def configure(platform, data_root=None):
    roots = {'colab': ('/content/water_segmentation_processed', '/content/drive/MyDrive/water_segmentation_v6'),
             'kaggle': (None, '/kaggle/working/water_segmentation_v6'),
             'local': (None, './water_segmentation_v6')}
    data, out = roots[platform]
    CFG['data'], CFG['out'] = data_root or data, out
    os.environ.setdefault('TORCH_HOME', '/tmp/torch_cache')
    os.environ.setdefault('HF_HOME', '/tmp/hf_cache')
    os.makedirs(CFG['out'], exist_ok=True)
    print(f'data={CFG["data"]}  out={CFG["out"]}  device={DEVICE}')


def bin_of(f):
    if f == 0:
        return '0%'
    for edge, label in EDGES:
        if f <= edge:
            return label
    return '>50%'


def patch_index(sensor):
    if sensor not in _INDEX:
        md, pd = (os.path.join(CFG['data'], sensor, d) for d in ('masks', 'patches'))
        items = []
        for f in sorted(os.listdir(md)):
            m = re.match(r'^(site_\d+_\w+)_y(\d{4})_m(\d{2})_patch_(\d+)\.npy$', f)
            if m:
                wf = float(np.load(os.path.join(md, f)).mean())
                items.append({'site': m[1], 'year': int(m[2]), 'month': int(m[3]), 'fname': f, 'wf': wf,
                              'bin': bin_of(wf), 'mask': os.path.join(md, f), 'patch': os.path.join(pd, f)})
        _INDEX[sensor] = items
        print(f'[index] {sensor}: {len(items)} patches')
    return _INDEX[sensor]


def val_site(test_site):
    return SITES[(SITES.index(test_site) + 1) % len(SITES)]


def in_season(p, season):
    return season == 'all' or p['month'] in SEASONS[season]


def split(sensor, test_site, train_season='all', test_season='all'):
    full, v = patch_index(sensor), val_site(test_site)
    tr = [p for p in full if p['site'] not in (test_site, v) and in_season(p, train_season)]
    va = [p for p in full if p['site'] == v]
    te = [p for p in full if p['site'] == test_site and in_season(p, test_season)]
    return tr, va, te, v


def training_pool(tr, rng, empty_ratio=0.25):
    ne = [p for p in tr if p['wf'] > 0]
    em = [p for p in tr if p['wf'] == 0]
    k = min(len(em), int(empty_ratio * len(ne)))
    return ne + [em[i] for i in rng.choice(len(em), k, replace=False)]


def norm_stats(pool, rng, n=500):
    sel = rng.choice(len(pool), min(n, len(pool)), replace=False)
    a = np.concatenate([np.nan_to_num(np.load(pool[i]['patch'])).reshape(9, -1) for i in sel], 1)
    s = a.std(1)
    return a.mean(1).astype('float32'), np.where(s < 1e-6, 1.0, s).astype('float32')


def sampler_weights(pool):
    counts = {}
    for p in pool:
        counts[p['bin']] = counts.get(p['bin'], 0) + 1
    nonempty = {b: n for b, n in counts.items() if b != '0%'}
    total = sum(nonempty.values())
    w = {b: total / n for b, n in nonempty.items()}
    w['0%'] = 0.5 * min(w.values())
    return torch.tensor([w[p['bin']] for p in pool], dtype=torch.double)


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(_):
    s = torch.initial_seed() % 2 ** 32
    np.random.seed(s); random.seed(s)


class PatchDS(Dataset):
    def __init__(self, items, mean, std, augment=False):
        self.items, self.m, self.s, self.aug = items, mean[:, None, None], std[:, None, None], augment

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        p = self.items[i]
        x = (np.nan_to_num(np.load(p['patch'])).astype('float32') - self.m) / self.s
        y = np.load(p['mask']).astype('float32')
        if self.aug:
            if np.random.rand() < 0.5:
                x, y = x[:, :, ::-1], y[:, ::-1]
            if np.random.rand() < 0.5:
                x, y = x[:, ::-1], y[::-1]
            k = np.random.randint(4)
            x, y = np.rot90(x, k, (1, 2)), np.rot90(y, k)
        return torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(np.ascontiguousarray(y))[None]


class BCEDice(nn.Module):
    def __init__(self, pos_weight=15.0):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=DEVICE))

    def forward(self, logits, y):
        p = torch.sigmoid(logits)
        dice = 1 - (2 * (p * y).sum((1, 2, 3)) + 1) / ((p + y).sum((1, 2, 3)) + 1)
        return 0.5 * self.bce(logits, y) + 0.5 * dice.mean()


def scores(tp, fp, fn):
    p, r = tp / (tp + fp + 1e-7), tp / (tp + fn + 1e-7)
    return {'iou': float(tp / (tp + fp + fn + 1e-7)), 'precision': float(p),
            'recall': float(r), 'f1': float(2 * p * r / (p + r + 1e-7))}


def counts(pred, y):
    return float((pred & y).sum()), float((pred & ~y).sum()), float((~pred & y).sum())


def best_threshold(values, y, grid):
    return float(max(grid, key=lambda t: scores(*counts(values > t, y))['iou']))


@torch.no_grad()
def predict(model, loader):
    model.eval()
    probs, ys = [], []
    for x, y in loader:
        probs.append(torch.sigmoid(model(x.to(DEVICE))).cpu())
        ys.append(y)
    return torch.cat(probs)[:, 0].numpy(), torch.cat(ys)[:, 0].numpy().astype(bool)


def out_paths(exp, seed, sensor, test_site):
    tag = f'{exp}_s{seed}'
    dirs = {k: os.path.join(CFG['out'], k, tag) for k in ('results', 'predictions', 'checkpoints')}
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    name = f'{sensor}_{test_site}'
    return (os.path.join(dirs['results'], name + '.json'), os.path.join(dirs['predictions'], name + '.npz'),
            os.path.join(dirs['checkpoints'], name + '.pt'))


def unet_fold(sensor, test_site, seed, exp, train_season='all', test_season='all', epochs=50,
              patience=15, pos_weight=15.0, batch_size=16, lr=1e-4, encoder_weights='imagenet', workers=2):
    import segmentation_models_pytorch as smp
    rpath, ppath, cpath = out_paths(exp, seed, sensor, test_site)
    if os.path.exists(rpath):
        return json.load(open(rpath))
    set_seed(seed)
    rng = np.random.default_rng(seed)
    tr, va, te, v = split(sensor, test_site, train_season, test_season)
    pool = training_pool(tr, rng)
    mean, std = norm_stats(pool, rng)
    g = torch.Generator().manual_seed(seed)
    kw = dict(batch_size=batch_size, num_workers=workers, pin_memory=True, worker_init_fn=seed_worker)
    train_dl = DataLoader(PatchDS(pool, mean, std, True), generator=g,
                          sampler=WeightedRandomSampler(sampler_weights(pool), len(pool), True, generator=g), **kw)
    val_dl, test_dl = (DataLoader(PatchDS(d, mean, std), shuffle=False, **kw) for d in (va, te))
    model = smp.Unet('resnet34', encoder_weights=encoder_weights, in_channels=9, classes=1).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=0.5, patience=7)
    crit = BCEDice(pos_weight)
    best, best_ep, wait, hist, t0 = -1.0, 0, 0, [], time.time()
    print(f'\n[{exp} s{seed}] {sensor} test={test_site} val={v} pool={len(pool)} test_n={len(te)}')
    for ep in range(1, epochs + 1):
        model.train()
        loss_sum = 0.0
        for x, y in train_dl:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
            loss_sum += loss.item()
        vp, vy = predict(model, val_dl)
        viou = scores(*counts(vp > 0.5, vy))['iou']
        sched.step(viou)
        hist.append({'epoch': ep, 'train_loss': loss_sum / len(train_dl), 'val_iou': viou})
        if ep == 1 or ep % 5 == 0:
            print(f'  ep {ep:02d} loss={loss_sum / len(train_dl):.4f} val_iou={viou:.4f}')
        if viou > best:
            best, best_ep, wait = viou, ep, 0
            torch.save(model.state_dict(), cpath)
        else:
            wait += 1
            if wait >= patience:
                break
    train_seconds = time.time() - t0
    model.load_state_dict(torch.load(cpath, map_location=DEVICE))
    vp, vy = predict(model, val_dl)
    thr = best_threshold(vp, vy, UNET_GRID)
    tp, ty = predict(model, test_dl)
    res = {'method': 'unet', 'sensor': sensor, 'test_site': test_site, 'val_site': v, 'seed': seed, 'exp': exp,
           'train_season': train_season, 'test_season': test_season, 'pool_size': len(pool), 'test_n': len(te),
           'best_epoch': best_ep, 'best_val_iou_at_0.5': best, 'threshold': thr,
           'threshold_at_grid_edge': thr in (float(UNET_GRID[0]), float(UNET_GRID[-1])),
           'val_at_threshold': scores(*counts(vp > thr, vy)), 'test': scores(*counts(tp > thr, ty)),
           'train_seconds': train_seconds, 'history': hist}
    np.savez_compressed(ppath, prob=(tp * 255).round().astype(np.uint8), fnames=np.array([p['fname'] for p in te]))
    json.dump(res, open(rpath, 'w'), indent=1)
    print(f'  TEST iou={res["test"]["iou"]:.4f} prec={res["test"]["precision"]:.4f} '
          f'rec={res["test"]["recall"]:.4f} thr={thr} time={train_seconds / 60:.1f} min')
    return res


def load_stack(items):
    return (np.stack([np.nan_to_num(np.load(p['patch'])).astype('float32') for p in items]),
            np.stack([np.load(p['mask']) for p in items]).astype(bool))


def classical_fold(sensor, test_site, seed=42, exp='classical', px_per_class=50, quantiles=200):
    rpath, ppath, _ = out_paths(exp, seed, sensor, test_site)
    if os.path.exists(rpath):
        return json.load(open(rpath))
    rng = np.random.default_rng(seed)
    tr, va, te, v = split(sensor, test_site)
    xv, yv = load_stack(va)
    xt, yt = load_stack(te)
    qgrid = lambda vals: np.quantile(vals, np.linspace(0.01, 0.999, quantiles))
    res = {'sensor': sensor, 'test_site': test_site, 'val_site': v, 'seed': seed, 'test_n': len(te)}
    for name, c in IDX.items():
        t = best_threshold(xv[:, c], yv, qgrid(xv[:, c]))
        res[name] = {**scores(*counts(xt[:, c] > t, yt)), 'threshold': t}
    Xs, ys = [], []
    for p in training_pool(tr, rng):
        x = np.nan_to_num(np.load(p['patch'])).reshape(9, -1).T
        m = np.load(p['mask']).ravel().astype(bool)
        for cls in (True, False):
            ix = np.flatnonzero(m == cls)
            if len(ix):
                ix = rng.choice(ix, min(px_per_class, len(ix)), replace=False)
                Xs.append(x[ix]); ys.append(np.full(len(ix), cls))
    t0 = time.time()
    rf = RandomForestClassifier(200, class_weight='balanced', n_jobs=-1, random_state=seed)
    rf.fit(np.concatenate(Xs), np.concatenate(ys))
    fit_seconds = time.time() - t0
    proba = lambda X: np.round(rf.predict_proba(X.transpose(0, 2, 3, 1).reshape(-1, 9))[:, 1], 6).reshape(X.shape[0], *X.shape[2:])
    pv, pt = proba(xv), proba(xt)
    t = best_threshold(pv, yv, qgrid(pv) + 5e-7)
    res['rf_default'] = scores(*counts(pt > 0.5, yt))
    res['rf_cal'] = {**scores(*counts(pt > t, yt)), 'threshold': t}
    res['rf_fit_seconds'] = fit_seconds
    np.savez_compressed(ppath, prob=(pt * 255).round().astype(np.uint8), pred=pt > t, fnames=np.array([p['fname'] for p in te]))
    json.dump(res, open(rpath, 'w'), indent=1)
    print(f'{sensor} {test_site} (val={v}): ' + '  '.join(
        f'{k}={res[k]["iou"]:.4f}' for k in [*IDX, 'rf_default', 'rf_cal']) + f'  rf_fit={fit_seconds:.0f}s')
    return res


def run_classical(sensors=('sentinel2', 'landsat')):
    for s in sensors:
        for site in SITES:
            classical_fold(s, site)


def run_unet(sensors, seeds, exp, train_season='all', test_season='all', **kw):
    for s in sensors:
        for seed in seeds:
            for site in SITES:
                unet_fold(s, site, seed, exp, train_season, test_season, **kw)


def status(exp, sensors, seeds):
    for s in sensors:
        for seed in seeds:
            done = [site for site in SITES if os.path.exists(out_paths(exp, seed, s, site)[0])]
            print(f'{exp} s{seed} {s}: {len(done)}/8 done' + ('' if len(done) == 8 else f'  missing={[x for x in SITES if x not in done]}'))


def summary(exp, sensor, seeds, key='test'):
    table = {}
    for site in SITES:
        vals = []
        for seed in seeds:
            p = out_paths(exp, seed, sensor, site)[0]
            if os.path.exists(p):
                vals.append(json.load(open(p))[key]['iou'])
        table[site] = vals
    print(f'\n{exp} {sensor}  seeds={list(seeds)}')
    fold_means = []
    for site, v in table.items():
        if v:
            fold_means.append(np.mean(v))
            print(f'  {site:<20} {np.mean(v):.4f} ± {np.std(v):.4f}  (n={len(v)})')
    if fold_means:
        print(f'  {"mean over sites":<20} {np.mean(fold_means):.4f} ± {np.std(fold_means):.4f}')
    return table
