#!/usr/bin/env python3

# Required python packages: requests pymupdf huggingface-hub levenshtein nltk

import requests

from typing import Optional, List
import os
import fitz
from pathlib import Path
import base64
import json

def rasterize_paper(
    pdf: Path,
    dpi: int = 96,
    pages=None,
) -> Optional[List[bytes]]:
    """
    Rasterize a PDF file to PNG images.

    Args:
        pdf (Path): The path to the PDF file.
        dpi (int, optional): The output DPI. Defaults to 96.
        pages (Optional[List[int]], optional): The pages to rasterize. If None, all pages will be rasterized. Defaults to None.

    Returns:
        Optional[List[io.bytes]]: The images as a list of bytes.
    """

    images = []
    if isinstance(pdf, (str, Path)):
        pdf = fitz.open(pdf)
    if pages is None:
        pages = range(len(pdf))
    for i in pages:
        page_bytes: bytes = pdf[i].get_pixmap(dpi=dpi).pil_tobytes(format="PNG")
        images.append(page_bytes)
    return images


def test_local(images):
    from huggingface_hub import snapshot_download
    from handler import handle
    class Ctx(object):
        pass
    ctx = Ctx()
    model_dir = snapshot_download(
        repo_id="facebook/nougat-small", local_dir_use_symlinks=False
    )

    ctx.manifest = {
        "createdOn": "04/01/2025 15:40:36",
        "runtime": "python",
        "model": {
            "modelName": "nougat_small",
            "handler": "handler.py",
            "modelFile": "model.safetensors",
            "modelVersion": "1.0"
        },
        "archiverVersion": "0.12.0"
    }
    ctx.system_properties = {
        "model_dir": model_dir,
        "gpu_id": "0"
    }
    req0 = {"data": json.dumps({"images": [base64.b64encode(image).decode("utf-8") for image in images[0:1]]})}
    req1 = {"data": json.dumps({"images": [base64.b64encode(image).decode("utf-8") for image in images[1:3]]})}
    resp_batch = handle([req0, req1],ctx)
    print("test_local responses:", resp_batch)

def test_remote(images):
    '''
    uv run create_mar.py
    torchserve --start --model-store ~/model-store --models nougat_small=nougat_small --ts-config config.properties --disable-token-auth --enable-model-api
    '''
    req0 = {"data": json.dumps({"images": [base64.b64encode(image).decode("utf-8") for image in images[0:1]]})}
    req1 = {"data": json.dumps({"images": [base64.b64encode(image).decode("utf-8") for image in images[1:3]]})}
    reqs = [req0, req1]
    for i in range(len(reqs)):
        resp = requests.post('http://localhost:8080/predictions/nougat_small', data=reqs[i])
        resp_data = json.loads(resp.content.decode("utf-8"))
        print(f"test_remote response {i}:", resp_data)

if __name__ == "__main__":
    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)
    pdf_path = os.path.join(script_dir, "nougat.pdf")
    BATCH_SIZE = 3 # extract text for only first 3 pages
    images = rasterize_paper(pdf=pdf_path, pages=range(BATCH_SIZE))
    # print(images)

    test_local(images)
    test_remote(images)

