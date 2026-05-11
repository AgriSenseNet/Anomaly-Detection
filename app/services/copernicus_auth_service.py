import os
import requests
from dotenv import load_dotenv

load_dotenv(".env.local")

TOKEN_URL = (
    "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/"
    "protocol/openid-connect/token"
)


def get_copernicus_access_token() -> str:
    data = {
        "client_id": "cdse-public",
        "username": os.getenv("COPERNICUS_USERNAME"),
        "password": os.getenv("COPERNICUS_PASSWORD"),
        "grant_type": "password",
    }

    totp = os.getenv("COPERNICUS_TOTP")
    if totp:
        data["totp"] = totp

    response = requests.post(TOKEN_URL, data=data, timeout=60)
    response.raise_for_status()

    return response.json()["access_token"]