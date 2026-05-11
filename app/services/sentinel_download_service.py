# from pathlib import Path
# import requests


# DOWNLOAD_BASE = "https://download.dataspace.copernicus.eu/odata/v1/Products"


# def download_product_zip(
#     product_id: str,
#     access_token: str,
#     output_dir: str = "downloads/sentinel2",
# ) -> Path:
#     """
#     Download one Sentinel-2 product ZIP from Copernicus Data Space.

#     WARNING:
#     Sentinel-2 products can be large.
#     Only run this after product search works.
#     """

#     output_path = Path(output_dir)
#     output_path.mkdir(parents=True, exist_ok=True)

#     zip_path = output_path / f"{product_id}.zip"

#     if zip_path.exists() and zip_path.stat().st_size > 0:
#         print(f"Product already downloaded: {zip_path}")
#         return zip_path

#     url = f"{DOWNLOAD_BASE}({product_id})/$value"

#     headers = {
#         "Authorization": f"Bearer {access_token}"
#     }

#     print(f"Downloading Sentinel-2 product to: {zip_path}")

#     with requests.get(url, headers=headers, stream=True, timeout=1200) as response:
#         response.raise_for_status()

#         with open(zip_path, "wb") as file:
#             for chunk in response.iter_content(chunk_size=1024 * 1024):
#                 if chunk:
#                     file.write(chunk)

#     print(f"Download completed: {zip_path}")

#     return zip_path

from pathlib import Path
from zipfile import ZipFile, is_zipfile
import requests


DOWNLOAD_BASE = "https://download.dataspace.copernicus.eu/odata/v1/Products"


def download_product_zip(
    product_id: str,
    access_token: str,
    output_dir: str = "downloads/sentinel2",
) -> Path:
    """
    Download one Sentinel-2 product ZIP and verify it is complete.
    Reuse existing verified ZIP if it already exists.
    """

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    zip_path = output_path / f"{product_id}.zip"
    temp_path = output_path / f"{product_id}.zip.part"

    # Reuse existing complete ZIP
    if zip_path.exists() and zip_path.stat().st_size > 0:
        if is_zipfile(zip_path):
            print(f"Product already downloaded and verified: {zip_path}")
            return zip_path

        print(f"Existing ZIP is invalid, deleting: {zip_path}")
        zip_path.unlink()

    # Remove old incomplete download
    if temp_path.exists():
        print(f"Removing incomplete temporary file: {temp_path}")
        temp_path.unlink()

    url = f"{DOWNLOAD_BASE}({product_id})/$value"

    headers = {
        "Authorization": f"Bearer {access_token}",
        "User-Agent": "smart-agri-c2-sentinel-pipeline/1.0",
    }

    print(f"Downloading from: {url}")
    print(f"Temporary file: {temp_path}")

    with requests.get(
        url,
        headers=headers,
        stream=True,
        timeout=(60, 1800),
        allow_redirects=True,
    ) as response:
        print("HTTP status:", response.status_code)
        print("Content-Type:", response.headers.get("Content-Type"))
        print("Content-Length:", response.headers.get("Content-Length"))
        print("Final URL:", response.url)

        response.raise_for_status()

        total_bytes = 0

        with open(temp_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    file.write(chunk)
                    total_bytes += len(chunk)

                    mb = total_bytes / (1024 * 1024)
                    print(f"\rDownloaded: {mb:.2f} MB", end="")

    print()

    if not temp_path.exists() or temp_path.stat().st_size == 0:
        raise RuntimeError("Download failed. File is empty.")

    print(f"Downloaded file size MB: {temp_path.stat().st_size / (1024 * 1024):.2f}")

    if not is_zipfile(temp_path):
        with open(temp_path, "rb") as file:
            preview = file.read(300)

        temp_path.unlink(missing_ok=True)

        raise RuntimeError(
            "Downloaded file is not a complete ZIP file. "
            f"First bytes: {preview!r}"
        )

    with ZipFile(temp_path, "r") as zip_file:
        bad_file = zip_file.testzip()

        if bad_file is not None:
            temp_path.unlink(missing_ok=True)
            raise RuntimeError(f"ZIP file is corrupted. First bad file: {bad_file}")

    temp_path.rename(zip_path)

    print(f"Download verified successfully: {zip_path}")

    return zip_path