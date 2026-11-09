import csv, json
import numpy as np
import pipeline as P
import analysis as A

HL = '/content/drive/MyDrive/hand_labels'
SITE = {'2': 'site_2_odo_otin', '5': 'site_5_ikere_gorge', '8': 'site_8_orelope'}
COMP = {'2': (2023, 2), '5': (2024, 11), '8': (2024, 3)}
METHODS = ['jrc'] + A.INDICES + ['rf_cal', 'unet_s0', 'unet_s1', 'unet_s2']
_cache = {}


def npz(exp, seed, sensor, site):
    k = (exp, seed, sensor, site)
    if k not in _cache:
        d = np.load(P.out_paths(exp, seed, sensor, site)[1])
        _cache[k] = (d, {f: i for i, f in enumerate(d['fnames'])})
    return _cache[k]


def locate(sensor, site, year, month, r, c):
    items = sorted([p for p in P.patch_index(sensor) if p['site'] == site and p['year'] == year and p['month'] == month],
                   key=lambda p: int(p['fname'].split('_patch_')[1][:-4]))
    return (items[(r // 128) * 4 + c // 128], r % 128, c % 128) if len(items) == 16 else None


def predict_point(sensor, site, p, pr, pc):
    cls = A.jload('classical', 42, sensor, site)
    x = np.nan_to_num(np.load(p['patch']))
    out = {m: bool(x[P.IDX[m], pr, pc] > cls[m]['threshold']) for m in A.INDICES}
    out['jrc'] = bool(np.load(p['mask'])[pr, pc])
    d, ix = npz('classical', 42, sensor, site)
    out['rf_cal'] = bool(d['pred'][ix[p['fname']]][pr, pc])
    for sd in (0, 1, 2):
        d, ix = npz('unet_all', sd, sensor, site)
        out[f'unet_s{sd}'] = bool(d['prob'][ix[p['fname']]][pr, pc] / 255 > A.jload('unet_all', sd, sensor, site)['threshold'])
    return out


def metrics(rows, m):
    tp = fp = fn = tn = 0.0
    for r in rows:
        pr, y, w = r['pred'][m], r['y'], r['w']
        tp += w * (pr and y); fp += w * (pr and not y); fn += w * (not pr and y); tn += w * (not pr and not y)
    nz = lambda a, b: a / b if b else float('nan')
    return {'iou': nz(tp, tp + fp + fn), 'precision': nz(tp, tp + fp), 'recall': nz(tp, tp + fn), 'oa': nz(tp + tn, tp + fp + fn + tn)}


def summarise(rows, m):
    if m == 'unet':
        ms = [metrics(rows, f'unet_s{s}') for s in range(3)]
        return {k: float(np.mean([x[k] for x in ms])) for k in ms[0]}
    return metrics(rows, m)


def run(n_boot=2000, seed=0):
    key = {r['id']: r for r in csv.DictReader(open(f'{HL}/key.csv'))}
    lab = {r['id'].strip(): r for r in csv.DictReader(open(f'{HL}/labels.csv', encoding='utf-8-sig')) if r['id'].strip()}
    areas = json.load(open(f'{HL}/strata_areas.json'))
    pts = [dict(key[i], y=l['label'].strip()) for i, l in lab.items() if l['label'].strip() in ('0', '1')]
    print(f'{len(pts)} usable labels; excluded {len(lab) - len(pts)} (unsure or blank)')
    groups = {}
    for p in pts:
        groups.setdefault((p['site'], p['stratum']), []).append(p)
    for (site, st), g in groups.items():
        for p in g:
            p['w'], p['y'] = areas[site][st] / len(g), p['y'] == '1'
    R = {}
    for sensor in ['sentinel2', 'landsat']:
        rows, missing = [], 0
        for p in pts:
            sid = p['id'][0]
            loc = locate(sensor, p['site'], *COMP[sid], int(p['row']), int(p['col']))
            if loc is None:
                missing += 1
                continue
            rows.append({**p, 'pred': predict_point(sensor, p['site'], *loc)})
        names = ['jrc'] + A.INDICES + ['rf_cal', 'unet']
        pooled = {m: summarise(rows, m) for m in names}
        rng = np.random.default_rng(seed)
        by_group = {}
        for r in rows:
            by_group.setdefault((r['site'], r['stratum']), []).append(r)
        boots = {m: [] for m in names}
        for _ in range(n_boot):
            bs = [g[i] for g in by_group.values() for i in rng.integers(0, len(g), len(g))]
            for m in names:
                boots[m].append(summarise(bs, m)['iou'])
        ci = {m: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))] for m, v in boots.items()}
        per_site = {s: {m: summarise([r for r in rows if r['site'] == s], m)['iou'] for m in names} for s in SITE.values()}
        R[sensor] = {'n_points': len(rows), 'missing_composite': missing, 'pooled': pooled, 'iou_ci95': ci, 'per_site_iou': per_site}
        print(f'\n=== {sensor.upper()} vs hand labels  ({len(rows)} points, {missing} without a composite)')
        print(f'{"method":<9}{"IoU":>7}{"95% CI":>16}{"prec":>7}{"recall":>8}{"OA":>7}   ' + '  '.join(s[5:11] for s in SITE.values()))
        for m in names:
            q = pooled[m]
            print(f'{m:<9}{q["iou"]:7.3f}   [{ci[m][0]:.3f}, {ci[m][1]:.3f}]{q["precision"]:7.3f}{q["recall"]:8.3f}{q["oa"]:7.3f}   '
                  + '  '.join(f'{per_site[s][m]:.3f}' for s in SITE.values()))
    json.dump(R, open(f'{HL}/hand_validation.json', 'w'), indent=1, default=float)
    return R
