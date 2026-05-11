from pathlib import Path
from zipfile import ZipFile
import tempfile

import rasterio


REQUIRED_BANDS = {
    "B02": None,
    "B03": None,
    "B04": None,
    "B08": None,
}


def find_band_files(extracted_dir: Path) -> dict:
    """
    Find Sentinel-2 10m band files inside extracted SAFE product.

    Required:
    - B02 blue
    - B03 green
    - B04 red
    - B08 near infrared
    """

    band_files = REQUIRED_BANDS.copy()

    jp2_files = list(extracted_dir.rglob("*.jp2"))

    for file_path in jp2_files:
        file_name = file_path.name

        for band in band_files:
            if f"_{band}_10m.jp2" in file_name:
                band_files[band] = file_path

    missing = [band for band, path in band_files.items() if path is None]

    if missing:
        raise FileNotFoundError(
            f"Missing Sentinel-2 band files: {missing}. "
            f"Found {len(jp2_files)} jp2 files."
        )

    return band_files


def read_band_array(file_path: Path):
    """
    Read one Sentinel-2 JP2 band as a numpy array.
    """

    with rasterio.open(file_path) as dataset:
        return dataset.read(1)


def read_required_bands_from_zip(zip_path: str | Path) -> dict:
    """
    Extract Sentinel-2 ZIP temporarily and read B02, B03, B04, B08 arrays.
    """

    zip_path = Path(zip_path)

    if not zip_path.exists():
        raise FileNotFoundError(f"ZIP file not found: {zip_path}")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        print(f"Extracting ZIP: {zip_path}")
        with ZipFile(zip_path, "r") as zip_file:
            zip_file.extractall(temp_path)

        print("Finding required Sentinel-2 bands...")
        band_files = find_band_files(temp_path)

        for band, path in band_files.items():
            print(f"{band}: {path.name}")

        return {
            "B02": read_band_array(band_files["B02"]),
            "B03": read_band_array(band_files["B03"]),
            "B04": read_band_array(band_files["B04"]),
            "B08": read_band_array(band_files["B08"]),
        }