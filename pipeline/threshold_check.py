import numpy as np
import pipeline as P
import analysis as A


def oracle_above_grid(sensors=('sentinel2', 'landsat'), seeds=(0, 1, 2)):
    out = {}
    for s in sensors:
        per_site = []
        for site in P.SITES:
            _, _, te, _ = P.split(s, site)
            Y = {p['fname']: np.load(p['mask']).astype(bool) for p in te}
            vals = []
            for sd in seeds:
                r = A.jload('unet_all', sd, s, site)
                u = np.load(P.out_paths('unet_all', sd, s, site)[1])
                pm, best = dict(zip(u['fnames'], u['prob'])), r['test']['iou']
                if r['threshold'] >= 0.99:
                    for k in (254, 255):
                        c = np.zeros(3)
                        for f, y in Y.items():
                            c += P.counts(pm[f] >= k, y)
                        best = max(best, P.scores(*c)['iou'])
                vals.append(best)
            per_site.append(np.mean(vals))
        out[s] = {'reported': float(A.site_scores(s, seeds)['unet'].mean()), 'oracle_upper_bound': float(np.mean(per_site))}
        print(f"{s}: reported {out[s]['reported']:.3f}, oracle upper bound above 0.99 = {out[s]['oracle_upper_bound']:.3f}")
    return out
