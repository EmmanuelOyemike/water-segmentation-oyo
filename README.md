# Surface water segmentation under geographic extrapolation

Code for *Surface Water Segmentation Under Geographic Extrapolation: A Cross-Sensor Benchmark of Spectral Indices, Random Forest and U-Net in Oyo State, Nigeria* (Oyemike, 2026).

The study compares NDWI, MNDWI and AWEI thresholds, a Random Forest and a U-Net (ResNet34 encoder) on Sentinel-2 and Landsat 8 monthly composites from eight sites in Oyo State (2022–2024), using leave-one-site-out testing with a rotating validation site, validation-calibrated thresholds for every method and three seeds for the U-Net.

## Repository layout

| Folder | Contents |
|---|---|
| `acquisition/` | Google Earth Engine export of monthly composites, JRC labels and JRC occurrence |
| `preprocessing/` | Tiling into 128 × 128 patches and the nine-channel feature stack |
| `pipeline/` | `pipeline.py` (data index, splits, methods, training), `analysis.py` (all post-hoc analyses), `label_sensitivity.py`, `threshold_check.py`, `make_hand_label_points.py`, `hand_validation.py` |
| `notebooks/` | Runner notebooks for each stage |
| `results/` | Result JSON files reported in the paper |
| `hand_labels/` | Independent point-based validation data |

## Reproducing the results

1. **Acquire data:** run the acquisition scripts with an Earth Engine project, exporting to Google Drive.
2. **Preprocess:** run `preprocessing/preprocess_multisite.py` to create `water_segmentation_processed/{sentinel2,landsat}/{patches,masks}`.
3. **Classical methods:** `notebooks/02_classical_and_seasonal.ipynb`, first two cells (CPU, about 30 minutes).
4. **All-season U-Net:** `notebooks/01_train_unet_kaggle.ipynb` as a committed Kaggle run (about 3.5 GPU hours).
5. **Seasonal design:** `notebooks/02_classical_and_seasonal.ipynb`, last cell (about 5–8 GPU hours).
6. **Analyses:** `notebooks/03_analysis.ipynb` (CPU).

All randomness is seeded. Folds already completed are skipped, so any stage can be resumed after a disconnect.

## Notes on implementation

- **Validation rotation:** for the test site at position *i*, the validation site is the site at position *i* + 1 (site 8 is validated on site 1).
- **Random Forest calibration:** forest probabilities are rounded to six decimals and candidate thresholds are offset by half a rounding step, so decisions do not depend on floating-point summation order. Exact predictions are saved alongside 8-bit probabilities.
- **U-Net predictions** are saved as 8-bit probabilities; the analysis scripts apply each model's validation threshold to them.

The published results were produced by an earlier revision of `classical_fold` followed by a Random Forest refit with the rounding rule above; the refit reproduced every stored Random Forest IoU exactly, and the current `classical_fold` applies the same rule directly. Early exploratory scripts and notebooks are not included.

## Requirements

See `requirements.txt`. Training used Kaggle and Google Colab GPUs (NVIDIA T4/P100); all analyses run on CPU.
