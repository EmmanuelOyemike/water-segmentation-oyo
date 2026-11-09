"""
Sensor-agnostic spectral index calculations.

Pass in green/red/nir/swir1 bands as numpy arrays (already reflectance-scaled,
0-1 range) regardless of which sensor they came from. This is what makes the
downstream ML pipeline sensor-agnostic: build a common feature space first.
"""

import numpy as np

EPS = 1e-6


def ndwi(green, nir):
    """McFeeters NDWI — water tends toward positive values."""
    return (green - nir) / (green + nir + EPS)


def mndwi(green, swir1):
    """Modified NDWI (Xu) — generally more robust in urban/built-up areas."""
    return (green - swir1) / (green + swir1 + EPS)


def ndvi(nir, red):
    """Used as an auxiliary feature to help suppress vegetation false positives."""
    return (nir - red) / (nir + red + EPS)


def awei(green, nir, swir1, blue=None):
    """Automated Water Extraction Index (simplified, no-shadow variant)."""
    if blue is None:
        return 4 * (green - swir1) - (0.25 * nir + 2.75 * swir1)
    return blue + 2.5 * green - 1.5 * (nir + swir1) - 0.25 * swir1


def build_feature_stack(bands: dict) -> np.ndarray:
    """
    bands: dict with keys 'blue','green','red','nir','swir1', each a 2D array
    (already normalized to reflectance). Returns an (H, W, C) stack of raw
    bands + computed indices, ready to feed to RF/XGBoost (flatten first) or
    as extra channels to the U-Net.
    """
    g, r, n, s = bands["green"], bands["red"], bands["nir"], bands["swir1"]
    features = [
        bands["blue"], g, r, n, s,
        ndwi(g, n),
        mndwi(g, s),
        ndvi(n, r),
        awei(g, n, s, bands.get("blue")),
    ]
    return np.stack(features, axis=-1)
