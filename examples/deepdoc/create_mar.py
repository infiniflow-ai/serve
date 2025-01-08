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
import glob
import pathlib
from huggingface_hub import snapshot_download
from model_archiver.model_packaging import generate_model_archive, ModelArchiverConfig

if len(sys.argv) != 2:
    print("Usage: uv run create_mar.py deviceIds")
    sys.exit(1)

all_device_types = ["cpu", "gpu"]
deviceType = "gpu"
deviceIds = None
if sys.argv[1] == "cpu":
    deviceType = "cpu"
else:
    deviceType = "gpu"
    deviceIds = sys.argv[1].split(",")
    deviceIds = [int(deviceId) for deviceId in deviceIds]

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

model_dir = snapshot_download(repo_id="InfiniFlow/deepdoc", local_dir_use_symlinks=False)
model_files = sorted(glob.glob(os.path.join(model_dir, "*.onnx"), recursive=False))
config_dict = {
    "model_name": "deepdoc",
    "handler": "handler.py",
    "version": "1.0",
    "serialized_file": None,
    "model_file": model_files[0],
    "extra_files": ",".join(model_files[1:]),
    "runtime": "python",
    "export_path": pathlib.Path.home() / "model-store",
    "archive_format": "no-archive",
    "force": True,
    "requirements_file": None,
    "config_file": "model_config.yaml",
}
config = ModelArchiverConfig(**config_dict)
generate_model_archive(config)
