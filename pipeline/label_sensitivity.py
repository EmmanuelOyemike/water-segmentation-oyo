import os, glob, json
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from scipy.stats import wilcoxon
import pipeline as P
from analysis import INDICES, jload

RAW = '/content/drive/MyDrive'


def template(site, sensor):
    hits = sorted(glob.glob(f'{RAW}/water_segmentation_monthly*/**/{site}_m*_{sensor}.tif', recursive=True))
    if not hits:
        raise FileNotFoundError(f'no composite found for {site} {sensor}')
    return hits[0]


def occurrence_on_grid(site, sensor, occ_dir):
    with rasterio.open(template(site, sensor)) as t:
        prof = t.profile
    out = np.zeros((prof['height'], prof['width']), np.float32)
    with rasterio.open(os.path.join(occ_dir, f'{site}_occurrence.tif')) as src:
        reproject(rasterio.band(src, 1), out, src_transform=src.transform, src_crs=src.crs,
                  dst_transform=prof['transform'], dst_crs=prof['crs'], resampling=Resampling.nearest)
    return out


def tile_lookup(items, grid):
    ncols, nrows = grid.shape[1] // 128, grid.shape[0] // 128
    groups = {}
    for p in items:
        groups.setdefault((p['year'], p['month']), []).append(p)
    lookup, skipped = {}, 0
    for g in groups.values():
        g = sorted(g, key=lambda p: int(p['fname'].split('_patch_')[1][:-4]))
        if len(g) != ncols * nrows:
            skipped += len(g)
            continue
        for rank, p in enumerate(g):
            r, c = divmod(rank, ncols)
            lookup[p['fname']] = grid[r * 128:(r + 1) * 128, c * 128:(c + 1) * 128]
    return lookup, skipped


def run_label_sensitivity(occ_dir=f'{RAW}/jrc_occurrence', seeds=(0, 1, 2), levels=(25, 50, 75),
                          sensors=('sentinel2', 'landsat')):
    R = {}
    for sensor in sensors:
        methods = INDICES + ['rf_cal'] + [f'unet_s{s}' for s in seeds]
        acc = {lv: {m: {s: np.zeros(3) for s in P.SITES} for m in methods} for lv in levels}
        agree = total = skipped = 0
        for site in P.SITES:
            _, _, te, _ = P.split(sensor, site)
            lookup, sk = tile_lookup(te, occurrence_on_grid(site, sensor, occ_dir))
            skipped += sk
            cls = jload('classical', 42, sensor, site)
            rf = np.load(P.out_paths('classical', 42, sensor, site)[1])
            probs = {'rf_cal': (dict(zip(rf['fnames'], rf['pred'])), cls['rf_cal']['threshold'])}
            for sd in seeds:
                u = np.load(P.out_paths('unet_all', sd, sensor, site)[1])
                probs[f'unet_s{sd}'] = (dict(zip(u['fnames'], u['prob'])), jload('unet_all', sd, sensor, site)['threshold'])
            for p in te:
                if p['fname'] not in lookup:
                    continue
                occ = lookup[p['fname']]
                total += 1
                agree += bool(np.array_equal(occ >= 50, np.load(p['mask']).astype(bool)))
                x = np.nan_to_num(np.load(p['patch']))
                preds = {m: x[P.IDX[m]] > cls[m]['threshold'] for m in INDICES}
                for m, (pm, thr) in probs.items():
                    preds[m] = pm[p['fname']] if m == 'rf_cal' else pm[p['fname']].astype(np.float32) / 255 > thr
                for lv in levels:
                    y = occ >= lv
                    for m, pr in preds.items():
                        acc[lv][m][site] += P.counts(pr, y)
        rs = {'mask_agreement_at_50': agree / max(total, 1), 'patches_used': total, 'patches_skipped': skipped, 'levels': {}}
        for lv in levels:
            site_iou = {m: np.array([P.scores(*acc[lv][m][s])['iou'] for s in P.SITES]) for m in methods}
            site_iou['unet'] = np.mean([site_iou[f'unet_s{s}'] for s in seeds], 0)
            means = {m: float(site_iou[m].mean()) for m in INDICES + ['rf_cal', 'unet']}
            w = wilcoxon(site_iou['rf_cal'], site_iou['unet'])
            rs['levels'][lv] = {'mean_iou': means, 'ranking': sorted(means, key=means.get, reverse=True),
                                'rf_cal_vs_unet_p': float(w.pvalue)}
        R[sensor] = rs
        print(f'\n=== {sensor.upper()}  mask agreement at 50% = {rs["mask_agreement_at_50"]:.3f} '
              f'({total} patches, {skipped} skipped)')
        print(f'{"method":<9}' + ''.join(f'{f">={lv}%":>9}' for lv in levels))
        for m in INDICES + ['rf_cal', 'unet']:
            print(f'{m:<9}' + ''.join(f'{rs["levels"][lv]["mean_iou"][m]:9.3f}' for lv in levels))
        for lv in levels:
            print(f'  >={lv}%: ranking {rs["levels"][lv]["ranking"]}  RF vs U-Net p={rs["levels"][lv]["rf_cal_vs_unet_p"]:.3f}')
    path = os.path.join(P.CFG['out'], 'analysis')
    os.makedirs(path, exist_ok=True)
    json.dump(R, open(os.path.join(path, 'label_sensitivity.json'), 'w'), indent=1, default=float)
    return R
