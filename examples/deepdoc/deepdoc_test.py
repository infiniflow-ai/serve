#!/usr/bin/env python

# Required python packages: requests opencv-python numpy huggingface-hub
import requests

import os
import base64
import json
import glob

import cv2
import numpy as np

def resize_image(image, new_width, new_height):
    # resize image
    resized_image = cv2.resize(
        image, (new_width, new_height), interpolation=cv2.INTER_LINEAR
    )

    # BGR to RGB
    rgb_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)

    # channel last (640, 640, 4) to channel first (3, 640, 640)
    rgb_image_channel_first = np.transpose(rgb_image, (2, 0, 1))

    # print image info
    print("channel last (H, W, C):", rgb_image.shape)
    print("channel first (C, H, W):", rgb_image_channel_first.shape)
    print("image data type:", rgb_image.dtype)
    return rgb_image_channel_first


def test_local(images):
    from huggingface_hub import snapshot_download
    from handler import handle

    class Ctx(object):
        pass

    ctx = Ctx()
    model_dir = snapshot_download(
        repo_id="InfiniFlow/deepdoc", local_dir_use_symlinks=False
    )

    ctx.manifest = {
        "createdOn": "04/01/2025 15:40:36",
        "runtime": "python",
        "model": {
            "modelName": "deepdoc",
            "handler": "handler.py",
            "modelFile": "",
            "modelVersion": "1.0",
        },
        "archiverVersion": "0.12.0",
    }
    ctx.system_properties = {"model_dir": model_dir, "gpu_id": "0"}
    req0 = {
        "data": json.dumps(
            {
                "model": "layout",
                "images": [
                    base64.b64encode(image.tobytes()).decode("utf-8") for image in images[0:1]
                ]
            }
        )
    }
    req1 = {
        "data": json.dumps(
            {
                "model": "layout",
                "images": [
                    base64.b64encode(image.tobytes()).decode("utf-8") for image in images[1:3]
                ]
            }
        )
    }
    resp_batch = handle([req0, req1], ctx)
    # print(resp_batch)
    responses = []
    for resp in resp_batch:
        rsp = []
        for boxes in resp:
            boxes = np.frombuffer(base64.b64decode(boxes.encode("utf-8")), dtype=np.float32).reshape(-1, 6)
            # print(boxes)
            rsp.append(boxes)
        responses.append(rsp)
    print("test_local responses:", responses)


def test_remote(images):
    """
    uv run create_mar.py 0
    torchserve --start --model-store ~/model-store --models deepdoc=deepdoc --ts-config config.properties --disable-token-auth --enable-model-api
    """
    req0 = {
        "data": json.dumps(
            {
                "model": "layout",
                "images": [
                    base64.b64encode(image.tobytes()).decode("utf-8") for image in images[0:1]
                ]
            }
        )
    }
    req1 = {
        "data": json.dumps(
            {
                "model": "layout",
                "images": [
                    base64.b64encode(image.tobytes()).decode("utf-8") for image in images[1:3]
                ]
            }
        )
    }
    reqs = [req0, req1]
    for i in range(len(reqs)):
        resp = requests.post("http://localhost:8080/predictions/deepdoc", data=reqs[i])
        rsp = []
        boxes_list = json.loads(resp.content.decode("utf-8"))
        for boxes in boxes_list:
            boxes = np.frombuffer(base64.b64decode(boxes.encode("utf-8")), dtype=np.float32).reshape(-1, 6)
            # print(boxes)
            rsp.append(boxes)
        print(f"test_remote response {i}:", rsp)


if __name__ == "__main__":
    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)
    image_paths = sorted(glob.glob(os.path.join(script_dir, "*.jpg")))
    images = []
    HEIGHT = WIDTH = 1024
    for image_path in image_paths:
        image = cv2.imread(image_path)
        assert image is not None
        resized_image = resize_image(image, HEIGHT, WIDTH)
        images.append(resized_image)

    test_local(images)
    test_remote(images)
