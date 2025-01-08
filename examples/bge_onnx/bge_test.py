#!/usr/bin/env python3

# PEP 723 metadata
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "huggingface-hub",
#   "levenshtein",
#   "nltk",
#   "numpy",
#   "onnxruntime-gpu",
#   "requests",
#   "transformers",
# ]
# ///

import os
import glob
import shutil
import base64
import json
import sys
import numpy as np
import requests


def test_local(repo_id, deviceType, deviceIds):
    from huggingface_hub import snapshot_download
    from handler import handle

    class Ctx(object):
        pass

    ctx = Ctx()
    model_name = repo_id.split("/")[-1]
    snapshot_dir = snapshot_download(
        repo_id=repo_id, local_dir_use_symlinks=False
    )
    model_dir = os.path.join(snapshot_dir, "onnx")
    onnx_model_fp = os.path.join(model_dir, "model.onnx")
    onnx_conf_fp = os.path.join(model_dir, "config.json")
    if not os.path.exists(onnx_model_fp):
        cmds = [
            "uv pip install optimum onnx onnxruntime",
            f"optimum-cli export onnx -m {repo_id} --opset 17 --optimize O2 --task feature-extraction --library-name transformers {model_dir}",
            ]
        print("Please run following commands to convert the model to onnx format at first:\n", "\n".join(cmds))
        sys.exit(1)
    if not os.path.exists(onnx_conf_fp):
        config_files = glob.glob(os.path.join(snapshot_dir, "*.json")) + glob.glob(os.path.join(snapshot_dir, "*.txt"))
        for config_file in config_files:
            shutil.copy(config_file, model_dir)

    ctx.manifest = {
        "createdOn": "04/01/2025 15:40:36",
        "runtime": "python",
        "model": {
            "modelName": model_name,
            "handler": "handler.py",
            "modelFile": "",
            "modelVersion": "1.0",
        },
        "archiverVersion": "0.12.0",
    }
    ctx.system_properties = {"model_dir": model_dir}
    if deviceType == "gpu":
        deviceId = str(deviceIds[0]) if deviceIds else "0"
        ctx.system_properties["gpu_id"] = deviceId
    req0 = {"data": json.dumps(["What is Deep Learning?"])}
    req1 = {"data": json.dumps(["What is Deep Learning?", "Hello world!"])}
    resp_batch = handle([req0, req1], ctx)
    # print(resp_batch)
    responses = []
    for resp in resp_batch:
        embeddings = []
        for embedding in resp:
            embedding = np.frombuffer(
                base64.b64decode(embedding.encode("utf-8")), dtype=np.float32
            )
            # print(embedding)
            embeddings.append(embedding)
        responses.append(embeddings)
    print("test_local responses:", responses)


def test_remote(repo_id):
    """
    Serving a single model:
    uv run create_mar.py BAAI/bge-m3 0
    torchserve --start --model-store ~/model-store --models bge-m3=bge-m3 --ts-config config.properties --disable-token-auth --enable-model-api

    Serving multiple models:
    torchserve --start --model-store ~/model-store --models all --ts-config config.properties --disable-token-auth --enable-model-api
    
    Stop serving:
    torchserve --stop
    """
    model_name = repo_id.split("/")[-1]
    req0 = {"data": json.dumps(["What is Deep Learning?"])}
    req1 = {"data": json.dumps(["What is Deep Learning?", "Hello world!"])}
    reqs = [req0, req1]
    for i in range(len(reqs)):
        resp = requests.post(f"http://localhost:8080/predictions/{model_name}", data=reqs[i])
        resp = json.loads(resp.content.decode("utf-8"))
        embeddings = []
        for embedding in resp:
            embedding =  np.frombuffer(
                base64.b64decode(embedding.encode("utf-8")), dtype=np.float32
            )
            embeddings.append(embedding)
        print(f"test_remote response {i}:", embeddings)


if __name__ == "__main__":
    args = sys.argv
    if len(args)<2 or len(args)>3:
        print("Usage: python3 bge_test.py <repo_id> [deviceIds]")
        sys.exit(1)
    repo_id = args[1]
    # TODO: https://huggingface.co/jinaai/jina-embeddings-v3
    # However, if you're working with the model directly, outside of the encode function, you'll need to apply mean pooling manually.
    # ValueError: Required inputs (['task_id']) are missing from input feed (['input_ids', 'attention_mask']).
    all_repo_ids = ["BAAI/bge-m3", "BAAI/bge-large-en-v1.5", "BAAI/bge-large-zh-v1.5", "jinaai/jina-embeddings-v3"]
    if repo_id not in all_repo_ids:
        print(f"repo_id shall be one of {all_repo_ids}")
        sys.exit(1)
    deviceType = "cpu"
    deviceIds = None
    if len(args) == 3:
        deviceType = "gpu"
        if "all" not in args[2]:
            deviceIds = args[2].split(",")
            deviceIds = [int(deviceId) for deviceId in deviceIds]
    test_local(repo_id, deviceType, deviceIds)
    test_remote(repo_id)
