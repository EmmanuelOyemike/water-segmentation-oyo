import os
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from indices import build_feature_stack

RAW_DIR = "data/raw/water_segmentation_monthly"
OUT_DIR = "data/processed"
PATCH_SIZE = 128
BAND_NAMES = ["blue", "green", "red", "nir", "swir1"]

SITES = [
    "site_1_oba", "site_2_odo_otin", "site_3_oyo_west", "site_4_egbeda",
    "site_5_ikere_gorge", "site_6_sepeteri", "site_7_kishi", "site_8_orelope",
]
MONTHS = [f"m{m:02d}" for m in range(1, 13) if m not in (7, 8)]


def load_and_align_label(label_path, target_profile):
    with rasterio.open(label_path) as src:
        label = np.zeros((target_profile["height"], target_profile["width"]), dtype="uint8")
        reproject(
            source=rasterio.band(src, 1), destination=label,
            src_transform=src.transform, src_crs=src.crs,
            dst_transform=target_profile["transform"], dst_crs=target_profile["crs"],
            resampling=Resampling.nearest,
        )
    return label


def tile_image_and_mask(image, mask, patch_size):
    _, h, w = image.shape
    for row in range(0, h - patch_size + 1, patch_size):
        for col in range(0, w - patch_size + 1, patch_size):
            yield (image[:, row:row + patch_size, col:col + patch_size],
                   mask[row:row + patch_size, col:col + patch_size])


def process_one_image(path, label_path, tag, patch_dir, mask_dir, start_idx):
    if not os.path.exists(path):
        return start_idx, 0

    with rasterio.open(path) as src:
        bands = {name: src.read(i + 1).astype("float32") for i, name in enumerate(BAND_NAMES)}
        profile = src.profile

    label = load_and_align_label(label_path, profile)
    feature_stack = build_feature_stack(bands).transpose(2, 0, 1)

    idx, n_saved = start_idx, 0
    for img_patch, mask_patch in tile_image_and_mask(feature_stack, label, PATCH_SIZE):
        if np.all(img_patch == 0):
            continue
        fname = f"{tag}_patch_{idx:04d}.npy"
        np.save(os.path.join(patch_dir, fname), img_patch.astype("float32"))
        np.save(os.path.join(mask_dir, fname), mask_patch.astype("uint8"))
        idx += 1
        n_saved += 1
    return idx, n_saved


def main():
    for sensor, suffix in [("sentinel2", "sentinel2"), ("landsat", "landsat")]:
        patch_dir = os.path.join(OUT_DIR, sensor, "patches")
        mask_dir  = os.path.join(OUT_DIR, sensor, "masks")
        os.makedirs(patch_dir, exist_ok=True)
        os.makedirs(mask_dir, exist_ok=True)

        idx = 0
        site_summary = {}

        for site in SITES:
            label_path = os.path.join(RAW_DIR, f"{site}_label.tif")
            if not os.path.exists(label_path):
                print(f"[SKIP] No label: {site}")
                continue

            site_patches = 0
            skipped = []
            for month in MONTHS:
                fname = f"{site}_{month}_{suffix}.tif"
                path  = os.path.join(RAW_DIR, fname)
                tag   = f"{site}_{month}"
                idx, n = process_one_image(path, label_path, tag, patch_dir, mask_dir, idx)
                if not os.path.exists(path):
                    skipped.append(month)
                else:
                    site_patches += n

            site_summary[site] = {"patches": site_patches, "skipped": skipped}
            skip_str = f"  [skipped {skipped}]" if skipped else ""
            print(f"  {site}: {site_patches} patches{skip_str}")

        print(f"\n{sensor}: {idx} total patches -> {patch_dir}")
        missing_months = {s: d["skipped"] for s, d in site_summary.items() if d["skipped"]}
        if missing_months:
            print(f"  Months skipped (missing files):")
            for s, ms in missing_months.items():
                print(f"    {s}: {ms}")


if __name__ == "__main__":
    main()