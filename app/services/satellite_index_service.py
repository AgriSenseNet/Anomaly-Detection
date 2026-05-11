import numpy as np


def safe_divide(numerator, denominator):
    """
    Divide safely without crashing when denominator is zero.
    """

    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype="float32"),
        where=denominator != 0,
    )


def compute_indices_from_arrays(blue_b02, green_b03, red_b04, nir_b08):
    """
    Compute NDVI, NDWI, EVI, and low NDVI percentage from Sentinel-2 bands.

    Bands:
    - B02 = blue
    - B03 = green
    - B04 = red
    - B08 = near infrared
    """

    blue = blue_b02.astype("float32")
    green = green_b03.astype("float32")
    red = red_b04.astype("float32")
    nir = nir_b08.astype("float32")

    ndvi = safe_divide(nir - red, nir + red)
    ndwi = safe_divide(green - nir, green + nir)
    evi = 2.5 * safe_divide(
        nir - red,
        nir + (6 * red) - (7.5 * blue) + 1,
    )

    ndvi = np.clip(ndvi, -1, 1)
    ndwi = np.clip(ndwi, -1, 1)
    evi = np.clip(evi, -1, 1)

    low_ndvi_zone_percent = float((ndvi < 0.30).mean() * 100)

    return {
        "ndvi_mean": float(np.nanmean(ndvi)),
        "ndwi_mean": float(np.nanmean(ndwi)),
        "evi_mean": float(np.nanmean(evi)),
        "low_ndvi_zone_percent": low_ndvi_zone_percent,
    }