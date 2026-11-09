# Independent point-based validation

Copy these files here from `MyDrive/hand_labels/`:

- `key.csv` — point ID, site, sampling stratum (JRC water / within 30 m of JRC water / land), grid row and column, coordinates
- `labels.csv` — the interpreter's labels (1 = water, 0 = not water, 9 = unsure) and the high-resolution image date per point
- `strata_areas.json` — pixel area of each stratum per site, used as estimation weights
- `points.kml` — the points as loaded into Google Earth Pro
- `hand_validation.json` — scored results

Points were labelled blind to `key.csv`.
