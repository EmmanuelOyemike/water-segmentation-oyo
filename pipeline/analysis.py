import os, json
import numpy as np
from scipy import ndimage
from scipy.stats import wilcoxon
import pipeline as P

INDICES = ['ndwi', 'mndwi', 'awei']
METHODS = INDICES + ['rf_cal', 'unet']
SEASON_EXPS = {('dry', 'dry'): 'unet_dry_dry', ('wet', 'dry'): 'unet_wet_dry',
               ('dry', 'wet'): 'unet_dry_wet', ('wet', 'wet'): 'unet_wet_wet'}
WIDTH_BINS = [(20, '<=20 m'), (50, '20-50 m'), (100, '50-100 m'), (300, '100-300 m'), (np.inf, '>300 m')]


def jload(exp, seed, sensor, site):
    return json.load(open(P.out_paths(exp, seed, sensor, site)[0]))


def site_scores(sensor, seeds, exp='unet_all'):
    cls = {s: jload('classical', 42, sensor, s) for s in P.SITES}
    out = {m: np.array([cls[s][m]['iou'] for s in P.SITES]) for m in INDICES + ['rf_default', 'rf_cal']}
    u = np.array([[jload(exp, sd, sensor, s)['test']['iou'] for sd in seeds] for s in P.SITES])
    out['unet'], out['unet_seed_sd'] = u.mean(1), u.std(1)
    return out


def describe(v):
    return {'mean': float(np.mean(v)), 'sd': float(np.std(v)), 'median': float(np.median(v)),
            'min': float(np.min(v)), 'max': float(np.max(v))}


def paired(a, b):
    d = np.asarray(a) - np.asarray(b)
    w = wilcoxon(a, b)
    return {'W': float(w.statistic), 'p': float(w.pvalue), 'wins_a': int((d > 0).sum()),
            'wins_b': int((d < 0).sum()), 'mean_diff': float(d.mean())}


def variance_components(Y):
    n, k = Y.shape
    g = Y.mean()
    ss_site = k * ((Y.mean(1) - g) ** 2).sum()
    ss_meth = n * ((Y.mean(0) - g) ** 2).sum()
    ss_tot = ((Y - g) ** 2).sum()
    ss_res = ss_tot - ss_site - ss_meth
    ms_site, ms_meth, ms_res = ss_site / (n - 1), ss_meth / (k - 1), ss_res / ((n - 1) * (k - 1))
    var_site = max((ms_site - ms_res) / k, 0.0)
    var_meth = max((ms_meth - ms_res) / n, 0.0)
    return {'share_site': ss_site / ss_tot, 'share_method': ss_meth / ss_tot, 'share_residual': ss_res / ss_tot,
            'var_site': var_site, 'var_method': var_meth, 'var_residual': ms_res,
            'icc_site': var_site / (var_site + ms_res) if var_site + ms_res > 0 else 0.0,
            'F_method': ms_meth / ms_res if ms_res > 0 else np.nan}


def bootstrap_components(Y, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    keys = ['share_site', 'share_method', 'icc_site']
    draws = {k: [] for k in keys}
    for _ in range(n_boot):
        vc = variance_components(Y[rng.integers(0, len(Y), len(Y))])
        for k in keys:
            draws[k].append(vc[k])
    return {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in draws.items()}


def width_bin(w):
    for edge, label in WIDTH_BINS:
        if w <= edge:
            return label


class Acc:
    def __init__(self):
        self.bins = {b: np.zeros(4) for b in P.BINS}
        self.agg30 = np.zeros(3)
        self.width = {lab: np.zeros(3) for _, lab in WIDTH_BINS}

    def add(self, pred, y, objects, bin_label):
        tp, fp, fn = P.counts(pred, y)
        self.bins[bin_label] += [tp, fp, fn, y.size]
        h, w = (y.shape[0] // 3) * 3, (y.shape[1] // 3) * 3
        blk = lambda a: a[:h, :w].reshape(h // 3, 3, w // 3, 3).mean((1, 3)) >= 0.5
        self.agg30 += P.counts(blk(pred), blk(y))
        lab, n, widths, areas = objects
        if n:
            hits = ndimage.sum(pred, lab, range(1, n + 1))
            for wd, ar, ht in zip(widths, areas, hits):
                self.width[width_bin(wd)] += [ht, ar, ht / ar >= 0.5]

    def summary(self):
        b = {k: ({'fpr': v[1] / v[3]} if k == '0%' else P.scores(*v[:3])) | {'n_patches': int(v[3] // (128 * 128))}
             for k, v in self.bins.items() if v[3]}
        wd = {k: {'pixel_recall': v[0] / v[1], 'object_detection_rate': None, 'pixels': int(v[1])}
              for k, v in self.width.items() if v[1]}
        return {'bins': b, 'iou_30m': P.scores(*self.agg30)['iou'], 'width': wd}


def objects_of(y, px_m=10):
    lab, n = ndimage.label(y, structure=np.ones((3, 3)))
    if not n:
        return lab, 0, [], []
    edt = ndimage.distance_transform_edt(y)
    idx = range(1, n + 1)
    widths = 2 * np.asarray(ndimage.maximum(edt, lab, idx)) * px_m
    areas = np.asarray(ndimage.sum(y, lab, idx))
    return lab, n, widths, areas


def coverage_analysis(sensor, seeds):
    accs = {m: Acc() for m in INDICES + ['rf_cal'] + [f'unet_s{s}' for s in seeds]}
    obj_counts = {m: {lab: [0, 0] for _, lab in WIDTH_BINS} for m in accs}
    for site in P.SITES:
        _, _, te, _ = P.split(sensor, site)
        cls = jload('classical', 42, sensor, site)
        probs = {}
        rf = np.load(P.out_paths('classical', 42, sensor, site)[1])
        probs['rf_cal'] = (dict(zip(rf['fnames'], rf['pred'])), cls['rf_cal']['threshold'])
        for sd in seeds:
            u = np.load(P.out_paths('unet_all', sd, sensor, site)[1])
            probs[f'unet_s{sd}'] = (dict(zip(u['fnames'], u['prob'])), jload('unet_all', sd, sensor, site)['threshold'])
        for p in te:
            x = np.nan_to_num(np.load(p['patch']))
            y = np.load(p['mask']).astype(bool)
            obj = objects_of(y)
            preds = {m: x[P.IDX[m]] > cls[m]['threshold'] for m in INDICES}
            for m, (pm, thr) in probs.items():
                preds[m] = pm[p['fname']] if m == 'rf_cal' else pm[p['fname']].astype(np.float32) / 255 > thr
            for m, pr in preds.items():
                accs[m].add(pr, y, obj, p['bin'])
                lab, n, widths, areas = obj
                if n:
                    hits = np.asarray(ndimage.sum(pr, lab, range(1, n + 1)))
                    for wd, ar, ht in zip(widths, areas, hits):
                        c = obj_counts[m][width_bin(wd)]
                        c[0] += int(ht / ar >= 0.5); c[1] += 1
    out = {}
    for m, a in accs.items():
        s = a.summary()
        for lab, (det, tot) in obj_counts[m].items():
            if lab in s['width']:
                s['width'][lab]['object_detection_rate'] = det / tot
                s['width'][lab]['n_objects'] = tot
        out[m] = s
    return out


def seed_mean(d, seeds, getter):
    vals = [getter(d[f'unet_s{s}']) for s in seeds]
    vals = [v for v in vals if v is not None]
    return (float(np.mean(vals)), float(np.std(vals))) if vals else (None, None)


def run_all(seeds=(0, 1, 2), sensors=('sentinel2', 'landsat'), n_boot=2000):
    R = {'main': {}, 'tests': {}, 'variance': {}, 'seasonal': {}, 'coverage': {}, 'timing': {}, 'thresholds': {}}
    for s in sensors:
        sc = site_scores(s, seeds)
        R['main'][s] = {m: describe(sc[m]) for m in INDICES + ['rf_default', 'rf_cal', 'unet']}
        R['main'][s]['per_site'] = {m: dict(zip(P.SITES, map(float, sc[m]))) for m in sc}
        best_idx = max(INDICES, key=lambda m: sc[m].mean())
        R['tests'][s] = {'rf_cal_vs_unet': paired(sc['rf_cal'], sc['unet']),
                         f'{best_idx}_vs_unet': paired(sc[best_idx], sc['unet']),
                         'rf_cal_vs_rf_default': paired(sc['rf_cal'], sc['rf_default'])}
        Y = np.column_stack([sc[m] for m in METHODS])
        R['variance'][s] = {**variance_components(Y), 'ci95': bootstrap_components(Y, n_boot),
                            'methods': METHODS}
        un = [jload('unet_all', sd, s, site) for sd in seeds for site in P.SITES]
        cl = [jload('classical', 42, s, site) for site in P.SITES]
        R['timing'][s] = {'unet_train_min_per_fold': float(np.mean([r['train_seconds'] for r in un]) / 60),
                          'rf_fit_s_per_fold': float(np.mean([r['rf_fit_seconds'] for r in cl]))}
        R['thresholds'][s] = {'unet_values': sorted(float(r['threshold']) for r in un),
                              'unet_at_grid_edge': int(sum(r['threshold_at_grid_edge'] for r in un)), 'n': len(un)}
        print(f'[coverage] {s} ...')
        R['coverage'][s] = coverage_analysis(s, seeds)
    cells = {k: np.array([[jload(e, sd, 'sentinel2', site)['test']['iou'] for sd in seeds] for site in P.SITES]).mean(1)
             for k, e in SEASON_EXPS.items()}
    R['seasonal'] = {'cells': {f'{a}->{b}': {**describe(v), 'per_site': dict(zip(P.SITES, map(float, v)))}
                               for (a, b), v in cells.items()},
                     'dry_test_train_effect': paired(cells[('dry', 'dry')], cells[('wet', 'dry')]),
                     'wet_test_train_effect': paired(cells[('dry', 'wet')], cells[('wet', 'wet')]),
                     'dry_models_test_season': paired(cells[('dry', 'dry')], cells[('dry', 'wet')]),
                     'wet_models_test_season': paired(cells[('wet', 'dry')], cells[('wet', 'wet')])}
    path = os.path.join(P.CFG['out'], 'analysis')
    os.makedirs(path, exist_ok=True)
    json.dump(R, open(os.path.join(path, 'analysis_results.json'), 'w'), indent=1, default=float)
    report(R, seeds)
    return R


def report(R, seeds):
    f = lambda v: '   -  ' if v is None else f'{v:6.3f}'
    for s, m in R['main'].items():
        print(f'\n=== {s.upper()} — LOSO pooled IoU across 8 sites')
        print(f'{"method":<11}{"mean":>8}{"sd":>8}{"median":>8}{"min":>8}{"max":>8}')
        for k in INDICES + ['rf_default', 'rf_cal', 'unet']:
            d = m[k]
            print(f'{k:<11}' + ''.join(f'{d[c]:8.3f}' for c in ['mean', 'sd', 'median', 'min', 'max']))
        print('U-Net seed SD per site: ' + ', '.join(f'{v:.3f}' for v in m['per_site']['unet_seed_sd'].values()))
        for k, t in R['tests'][s].items():
            print(f'  {k:<22} W={t["W"]:<5} p={t["p"]:.3f}  wins {t["wins_a"]}-{t["wins_b"]}  mean diff={t["mean_diff"]:+.3f}')
        v = R['variance'][s]
        ci = lambda k: '[' + ', '.join(f'{x:.2f}' for x in v['ci95'][k]) + ']'
        print(f'  variance shares: site={v["share_site"]:.2f} {ci("share_site")}  method={v["share_method"]:.2f} '
              f'{ci("share_method")}  residual={v["share_residual"]:.2f}')
        print(f'  ICC(site)={v["icc_site"]:.2f} {ci("icc_site")}  F(method)={v["F_method"]:.2f}')
        print(f'  timing: U-Net {R["timing"][s]["unet_train_min_per_fold"]:.1f} min/fold, RF fit {R["timing"][s]["rf_fit_s_per_fold"]:.0f} s/fold')
        th = R['thresholds'][s]
        print(f'  U-Net thresholds at grid edge: {th["unet_at_grid_edge"]}/{th["n"]}  range {th["unet_values"][0]}-{th["unet_values"][-1]}')
        cov = R['coverage'][s]
        print(f'  {"by water fraction":<12}' + ''.join(f'{b:>10}' for b in P.BINS) + f'{"IoU@30m":>10}')
        rows = INDICES + ['rf_cal']
        for k in rows:
            bins = cov[k]['bins']
            print(f'  {k:<12}' + ''.join(f(bins[b].get('fpr', bins[b].get('iou'))) .rjust(10) if b in bins else '        - '
                                         for b in P.BINS) + f(cov[k]['iou_30m']).rjust(10))
        ub = {b: seed_mean(cov, seeds, lambda d: d['bins'].get(b, {}).get('fpr', d['bins'].get(b, {}).get('iou'))) for b in P.BINS}
        u30 = seed_mean(cov, seeds, lambda d: d['iou_30m'])
        print(f'  {"unet":<12}' + ''.join(f(ub[b][0]).rjust(10) for b in P.BINS) + f(u30[0]).rjust(10))
        print(f'  {"width (recall)":<16}' + ''.join(f'{lab:>11}' for _, lab in WIDTH_BINS))
        for k in rows:
            w = cov[k]['width']
            print(f'  {k:<16}' + ''.join(f(w[lab]['pixel_recall'] if lab in w else None).rjust(11) for _, lab in WIDTH_BINS))
        uw = [seed_mean(cov, seeds, lambda d, lab=lab: d['width'].get(lab, {}).get('pixel_recall'))[0] for _, lab in WIDTH_BINS]
        print(f'  {"unet":<16}' + ''.join(f(v).rjust(11) for v in uw))
    print('\n=== SEASONAL (Sentinel-2 U-Net, seed means)')
    for k, c in R['seasonal']['cells'].items():
        print(f'  {k:<9} mean={c["mean"]:.3f} sd={c["sd"]:.3f}')
    for k in ['dry_test_train_effect', 'wet_test_train_effect', 'dry_models_test_season', 'wet_models_test_season']:
        t = R['seasonal'][k]
        print(f'  {k:<24} W={t["W"]:<5} p={t["p"]:.3f}  wins {t["wins_a"]}-{t["wins_b"]}  mean diff={t["mean_diff"]:+.3f}')
