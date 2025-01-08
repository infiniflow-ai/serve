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
import os
import pathlib
import shutil
from huggingface_hub import snapshot_download
from model_archiver.model_packaging import ModelArchiverConfig, generate_model_archive

if len(sys.argv) != 3:
    print("Usage: uv run create_mar.py <repo_id> deviceIds")
    sys.exit(1)

repo_id = sys.argv[1]
all_repo_ids = ["BAAI/bge-m3", "BAAI/bge-large-zh-v1.5", "BAAI/bge-large-en-v1.5", "maidalun1020/bce-embedding-base_v1"]
if repo_id not in all_repo_ids:
    print(f"repo_id shall be one of {all_repo_ids}")
    sys.exit(1)
deviceType = "gpu"
deviceIds = sys.argv[2].split(",")
deviceIds = [int(deviceId) for deviceId in deviceIds]

model_name = repo_id.split("/")[-1]
snapshot_dir = snapshot_download(repo_id=repo_id, local_dir_use_symlinks=False)

model_config = """# model specific config, refers to model-archiver/README.md
batchSize: 8 # default: 1
"""
if deviceType == "gpu":
    model_config += "deviceType: gpu # cpu, gpu, neuron\n"
    if deviceIds is not None:
        model_config += f"deviceIds: {deviceIds} # gpu device ids allocated to this model.\n"
        model_config += f"minWorkers: {len(deviceIds)}\n"
        model_config += f"maxWorkers: {len(deviceIds)}\n"
else:
    model_config += "deviceType: cpu # cpu, gpu, neuron\n"
script_path = os.path.abspath(__file__)
config_file = os.path.join(os.path.dirname(script_path), "model_config.yaml")
with open(config_file, "w") as f:
    f.write(model_config)

# It doesn't matter if extra_files includes the model file.
extra_files = [str(item) for item in pathlib.Path(snapshot_dir).iterdir() if item.is_file() and 'onnx' not in item.name]
config_dict = {
    "model_name": model_name,
    "handler": "handler.py",
    "version": "1.0",
    "serialized_file": None,
    "model_file": None,
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

# vLLM need 1_Pooling however generate_model_archive doesn't keep directory hierarchy.
extra_dirs = [item for item in pathlib.Path(snapshot_dir).iterdir() if item.is_dir() and item.name != "onnx"]
model_dir = pathlib.Path.home() / "model-store" / model_name
print(f"copying {extra_dirs} to {model_dir}")
for extra_dir in extra_dirs:
    shutil.copytree(str(extra_dir), str(model_dir / extra_dir.name), dirs_exist_ok=True)
