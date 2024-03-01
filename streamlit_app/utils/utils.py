import requests
from requests_toolbelt.multipart.encoder import MultipartEncoder


def process(image, server_url: str):
    file = MultipartEncoder(fields={'file': ('filename', image, 'image/jpeg')})

    response = requests.post(
        server_url, data=file, headers={'Content-Type': file.content_type}, timeout=8000
    )

    return response
