#!/usr/bin/env python3

# PEP 723 metadata
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "levenshtein",
#   "nltk",
#   "numpy",
#   "huggingface-hub",
#   "requests",
#   "transformers",
# ]
# ///

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
    model_dir = snapshot_dir

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
    else:
        # vllm requires gpu
        assert deviceType == "gpu"
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
    uv pip install huggingface-hub torch-model-archiver torchserve nvgpu
    uv run create_mar.py BAAI/bge-m3 0
    torchserve --start --model-store ~/model-store --models bge-m3=bge-m3 --ts-config config.properties --disable-token-auth --enable-model-api

    Serving multiple models:
    torchserve --start --model-store ~/model-store --models all --ts-config config.properties --disable-token-auth --enable-model-api
    
    Stop serving:
    torchserve --stop
    """
    model_name = repo_id.split("/")[-1]
    req0 = {"data": json.dumps(["""9.13.1   Licensed corporations and registered institutions are primarily  responsible for planning and implementing a continuous education  programme best suited to the training needs of the licensed  representatives or relevant individuals they engage. Such programmes  should enhance the individuals’industry knowledge, skills and  professionalism. The firms should perform due diligence to ensure CPT  compliance by the individuals they engage.  
  9.13.2   Licensed individuals and relevant individuals of registered institutions are required to complete 10 CPT hours per calendar year, regardless of  the number and types of regulated activities he or she engages in. Five of these 10 CPT hours must be on topics directly relevant to the  regulated activities for which he or she is licensed at the time the CPT  hours are undertaken.   
  9.13.3   Individuals who engage in the sponsor work or Codes on Takeovers  transaction work for a firm are required to attend 2.5 CPT hours per  calendar year on topics that are relevant to their sponsor work or Codes  on Takeovers advisory work.  
  9.13.4   In view of the higher level of responsibility and accountability placed on  Responsible officers and Executive Officers, they are required to take  two additional CPT hours per calendar year on regulatory compliance.  
  9.13.5   Within the 12 months after a person first becomes a licensed individual  or relevant individuals, he or she must undertake two CPT hours on  ethics. Thereafter, that person is required to complete two CPT hours  per calendar year on topics relating to either ethics or compliance.  
  9.13.6   Details of CPT requirements for corporations and individuals are set out  in paragraphs 4 and 5 of the “Guidelines on Continuous Professional  Training"""])}
    req1 = {"data": json.dumps(["What is Deep Learning?", "Hello world!"])}
    reqs = [req0, req1]
    for i in range(len(reqs)):
        resp = requests.post(f"http://192.168.0.252:8080/predictions/{model_name}", data=reqs[i])
        # resp = requests.post(f"http://192.168.200.181:4570/predictions/{model_name}", data=reqs[i])
        resp = json.loads(resp.content.decode("utf-8"))
        if not isinstance(resp, list):
            print(f"test_remote unexpected response body: {resp}")
            continue
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
    all_repo_ids = ["BAAI/bge-m3", "BAAI/bge-large-en-v1.5", "BAAI/bge-large-zh-v1.5", "maidalun1020/bce-embedding-base_v1"]
    if repo_id not in all_repo_ids:
        print(f"repo_id shall be one of {all_repo_ids}")
        sys.exit(1)
    deviceType = "gpu"
    deviceIds = None
    if len(args) == 3:
        if "all" not in args[2]:
            deviceIds = args[2].split(",")
            deviceIds = [int(deviceId) for deviceId in deviceIds]
    # test_local(repo_id, deviceType, deviceIds)
    test_remote(repo_id)
