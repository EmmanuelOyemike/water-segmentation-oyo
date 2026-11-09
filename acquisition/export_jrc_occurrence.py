import ee

PROJECT_ID = "water-segmentation-504623"
SITES = {
    "site_1_oba": (4.1981, 8.1778), "site_2_odo_otin": (4.5528, 7.9641),
    "site_3_oyo_west": (3.9027, 7.8895), "site_4_egbeda": (4.1294, 7.4097),
    "site_5_ikere_gorge": (3.7573, 8.2078), "site_6_sepeteri": (3.6332, 8.5780),
    "site_7_kishi": (3.8440, 9.0535), "site_8_orelope": (3.7601, 8.8141),
}

ee.Initialize(project=PROJECT_ID)
occ = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("occurrence").unmask(0).toUint8()
for name, (lon, lat) in SITES.items():
    aoi = ee.Geometry.Point([lon, lat]).buffer(3000).bounds()
    ee.batch.Export.image.toDrive(image=occ.clip(aoi), description=f"{name}_occurrence", folder="jrc_occurrence",
                                  region=aoi, scale=10, maxPixels=1e9, fileFormat="GeoTIFF").start()
    print("started", name)
