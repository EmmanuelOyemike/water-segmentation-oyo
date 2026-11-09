import os, csv, json
import numpy as np
import rasterio
from rasterio.warp import transform as wtransform
from scipy import ndimage
import label_sensitivity as L

SITES = ['site_2_odo_otin', 'site_5_ikere_gorge', 'site_8_orelope']


def make_points(occ_dir, out='/content/drive/MyDrive/hand_labels', n=40, seed=7):
    os.makedirs(out, exist_ok=True)
    rng, rows, areas = np.random.default_rng(seed), [], {}
    for site in SITES:
        occ = L.occurrence_on_grid(site, 'sentinel2', occ_dir)[:512, :512]
        w = occ >= 50
        edge = ndimage.binary_dilation(w, iterations=3) & ~w
        strata = {'water': w, 'edge': edge, 'land': ~w & ~edge}
        areas[site] = {k: int(m.sum()) for k, m in strata.items()}
        with rasterio.open(L.template(site, 'sentinel2')) as t:
            tr, crs = t.transform, t.crs
        site_rows = []
        for name, m in strata.items():
            rr, cc = np.nonzero(m)
            for i in rng.choice(len(rr), min(n, len(rr)), replace=False):
                x, y = rasterio.transform.xy(tr, rr[i], cc[i])
                lon, lat = wtransform(crs, 'EPSG:4326', [x], [y])
                site_rows.append({'site': site, 'stratum': name, 'row': int(rr[i]), 'col': int(cc[i]),
                                  'lon': round(lon[0], 6), 'lat': round(lat[0], 6)})
        rng.shuffle(site_rows)
        for j, p in enumerate(site_rows):
            p['id'] = f'{site[5]}-{j + 1:03d}'
        rows += site_rows
    csv.writer(open(f'{out}/key.csv', 'w', newline='')).writerows([list(rows[0])] + [list(p.values()) for p in rows])
    csv.writer(open(f'{out}/labels_to_fill.csv', 'w', newline='')).writerows(
        [['id', 'lon', 'lat', 'label', 'image_date']] + [[p['id'], p['lon'], p['lat'], '', ''] for p in rows])
    kml = ['<?xml version="1.0" encoding="UTF-8"?><kml xmlns="http://www.opengis.net/kml/2.2"><Document>']
    kml += [f'<Placemark><name>{p["id"]}</name><Point><coordinates>{p["lon"]},{p["lat"]},0</coordinates></Point></Placemark>' for p in rows]
    open(f'{out}/points.kml', 'w').write('\n'.join(kml + ['</Document></kml>']))
    json.dump(areas, open(f'{out}/strata_areas.json', 'w'))
    print(len(rows), 'points written to', out)
