"""
Export monthly median composites for Landsat and Sentinel-2 across all
8 verified Oyo State sites. Replaces the paired-date approach with monthly
composites that aggregate all cloud-free observations within each month,
producing more data and enabling proper seasonal analysis.

Usage:
    python src/data_acquisition_monthly.py --year 2022
    python src/data_acquisition_monthly.py --year 2023
    python src/data_acquisition_monthly.py --year 2024

Exports to Google Drive folder: water_segmentation_monthly_{year}
Naming convention: {site_name}_m{month:02d}_{sensor}
e.g. site_1_oba_m01_sentinel2, site_5_ikere_gorge_m07_landsat

After all tasks complete, download from Google Drive and run
preprocess_multisite.py as before.
"""

import ee
import argparse

PROJECT_ID = "water-segmentation-504623"

SITES = {
    "site_1_oba": {
        "lon": 4.1981, "lat": 8.1778, "half_width_km": 3.0,
        "description": "River Oba, Surulere LGA",
    },
    "site_2_odo_otin": {
        "lon": 4.5528, "lat": 7.9641, "half_width_km": 3.0,
        "description": "Odo Otin, straddles Oyo/Osun border - documented",
    },
    "site_3_oyo_west": {
        "lon": 3.9027, "lat": 7.8895, "half_width_km": 3.0,
        "description": "Oyo West LGA water body",
    },
    "site_4_egbeda": {
        "lon": 4.1294, "lat": 7.4097, "half_width_km": 3.0,
        "description": "Egbeda LGA water body",
    },
    "site_5_ikere_gorge": {
        "lon": 3.7573, "lat": 8.2078, "half_width_km": 3.0,
        "description": "Ikere Gorge Dam reservoir, Itesiwaju LGA - JRC 95.5%",
    },
    "site_6_sepeteri": {
        "lon": 3.6332, "lat": 8.5780, "half_width_km": 3.0,
        "description": "Sepeteri water body, Saki East LGA - JRC 70.2%",
    },
    "site_7_kishi": {
        "lon": 3.8440, "lat": 9.0535, "half_width_km": 3.0,
        "description": "Kishi area, Irepo LGA - JRC 66.3% seasonal",
    },
    "site_8_orelope": {
        "lon": 3.7601, "lat": 8.8141, "half_width_km": 3.0,
        "description": "Orelope LGA water body - JRC 50.3% seasonal",
    },
}

MONTHS = list(range(1, 13))

LANDSAT_COLLECTION = "LANDSAT/LC08/C02/T1_L2"
S2_CLOUD_PROB_THRESHOLD = 35

S2_BANDS = {"B2": "blue", "B3": "green", "B4": "red",
            "B8": "nir",  "B11": "swir1"}
LS_BANDS = {"SR_B2": "blue", "SR_B3": "green", "SR_B4": "red",
            "SR_B5": "nir",  "SR_B6": "swir1"}


def build_aoi(site):
    return (
        ee.Geometry.Point([site["lon"], site["lat"]])
        .buffer(site["half_width_km"] * 1000)
        .bounds()
    )


def mask_s2_clouds(img):
    cloud_prob = img.select("MSK_CLDPRB")
    return img.updateMask(cloud_prob.lt(S2_CLOUD_PROB_THRESHOLD))


def get_s2_monthly_composite(aoi, year, month):
    start = ee.Date.fromYMD(year, month, 1)
    end   = start.advance(1, "month")
    return (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(aoi)
        .filterDate(start, end)
        .map(mask_s2_clouds)
        .select(list(S2_BANDS.keys()))
        .median()
        .rename(list(S2_BANDS.values()))
        .clip(aoi)
    )


def mask_landsat_clouds(img):
    qa = img.select("QA_PIXEL")
    clear = (qa.bitwiseAnd(1 << 3).eq(0)
               .And(qa.bitwiseAnd(1 << 4).eq(0)))
    return img.updateMask(clear)


def get_landsat_monthly_composite(aoi, year, month):
    start = ee.Date.fromYMD(year, month, 1)
    end   = start.advance(1, "month")

    def scale_sr(img):
        optical = img.select("SR_B.").multiply(0.0000275).add(-0.2)
        return img.addBands(optical, None, True)

    return (
        ee.ImageCollection(LANDSAT_COLLECTION)
        .filterBounds(aoi)
        .filterDate(start, end)
        .map(mask_landsat_clouds)
        .map(scale_sr)
        .select(list(LS_BANDS.keys()))
        .median()
        .rename(list(LS_BANDS.values()))
        .clip(aoi)
    )


def get_water_label(aoi):
    jrc = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("occurrence")
    return jrc.gte(50).rename("water_label").clip(aoi)


def export_image(image, region, description, scale=10, folder="water_segmentation_monthly"):
    task = ee.batch.Export.image.toDrive(
        image=image,
        description=description,
        folder=folder,
        region=region,
        scale=scale,
        maxPixels=1e9,
        fileFormat="GeoTIFF",
    )
    task.start()
    print(f"    Started: {description}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, required=True,
                        help="Year for monthly composites (e.g. 2022, 2023, 2024)")
    parser.add_argument("--skip-labels", action="store_true",
                        help="Skip label exports (JRC label is time-invariant; only needed once)")
    args = parser.parse_args()

    year = args.year
    drive_folder = f"water_segmentation_monthly_{year}"

    ee.Initialize(project=PROJECT_ID)

    print(f"\nExporting year: {year}")
    print(f"Drive folder:  {drive_folder}")
    print(f"Skip labels:   {args.skip_labels}\n")

    total_exports = 0

    for site_name, site in SITES.items():
        print(f"\n=== {site_name} ({site['description']}) ===")
        aoi = build_aoi(site)

        if not args.skip_labels:
            label = get_water_label(aoi)
            export_image(label, aoi, f"{site_name}_label",
                         scale=10, folder=drive_folder)
            total_exports += 1

        for month in MONTHS:
            month_tag = f"m{month:02d}"

            s2_desc = f"{site_name}_{month_tag}_sentinel2"
            export_image(get_s2_monthly_composite(aoi, year, month),
                         aoi, s2_desc, scale=10, folder=drive_folder)
            total_exports += 1

            ls_desc = f"{site_name}_{month_tag}_landsat"
            export_image(get_landsat_monthly_composite(aoi, year, month),
                         aoi, ls_desc, scale=10, folder=drive_folder)
            total_exports += 1

    print(f"\nAll export tasks submitted.")
    print(f"Total exports this run: {total_exports}")
    print(f"Check progress: code.earthengine.google.com/tasks")
    print(f"Download from Drive folder: {drive_folder}")