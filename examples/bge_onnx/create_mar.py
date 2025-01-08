#!/usr/bin/env python3

# PEP 723 metadata
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "huggingface-hub",
#   "torch-model-archiver",
# ]
# ///

import sys
import shutil
import os
import glob
import pathlib
from huggingface_hub import snapshot_download
from model_archiver.model_packaging import generate_model_archive, ModelArchiverConfig

if len(sys.argv) <2 or len(sys.argv)> 3:
    print("Usage: uv run create_mar.py <repo_id> [deviceIds]")
    sys.exit(1)

repo_id = sys.argv[1]
all_repo_ids = ["BAAI/bge-m3", "BAAI/bge-large-zh-v1.5", "BAAI/bge-large-en-v1.5", "jinaai/jina-embeddings-v3"]
all_device_types = ["cpu", "gpu"]
if repo_id not in all_repo_ids:
    print(f"repo_id shall be one of {all_repo_ids}")
    sys.exit(1)
deviceType = "cpu"
deviceIds = None
if len(sys.argv) == 3:
    deviceType = "gpu"
    if "all" not in sys.argv[2]:
        deviceIds = sys.argv[2].split(",")
        deviceIds = [int(deviceId) for deviceId in deviceIds]

model_name = repo_id.split("/")[-1]
snapshot_dir = snapshot_download(repo_id=repo_id, local_dir_use_symlinks=False)
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

model_config = """# model specific config, refers to model-archiver/README.md
batchSize: 3 # default: 1
"""
if deviceType == "gpu":
    model_config += "deviceType: gpu # cpu, gpu, neuron\n"
    if deviceIds is not None:
        model_config += f"deviceIds: {deviceIds} # gpu device ids allocated to this model.\n"
else:
    model_config += "deviceType: cpu # cpu, gpu, neuron\n"
script_path = os.path.abspath(__file__)
config_file = os.path.join(os.path.dirname(script_path), "model_config.yaml")
with open(config_file, "w") as f:
    f.write(model_config)

# It doesn't matter if extra_files includes the model file.
extra_files = sorted(glob.glob(os.path.join(model_dir, "*"), recursive=False))
config_dict = {
    "model_name": model_name,
    "handler": "handler.py",
    "version": "1.0",
    "serialized_file": None,
    "model_file": onnx_model_fp,
    "extra_files": ",".join(extra_files),
    "runtime": "python",
    "export_path": pathlib.Path.home() / "model-store",
    "archive_format": "no-archive",
    "force": True,
    "requirements_file": None,
    "config_file": "model_config.yaml",
}
config = ModelArchiverConfig(**config_dict)
generate_model_archive(config)
