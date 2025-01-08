#!/usr/bin/env python3

import tempfile
import os
import pathlib
from transformers import AutoProcessor, VisionEncoderDecoderModel
from model_archiver.model_packaging import generate_model_archive, ModelArchiverConfig

processor = AutoProcessor.from_pretrained("facebook/nougat-small")
model = VisionEncoderDecoderModel.from_pretrained("facebook/nougat-small")

with tempfile.TemporaryDirectory() as fp:
    processor.save_pretrained(fp)
    model.save_pretrained(fp)
    model_file = os.path.join(fp, "model.safetensors")
    extra_files  = "config.json generation_config.json preprocessor_config.json special_tokens_map.json tokenizer_config.json tokenizer.json".split()
    extra_files = [os.path.join(fp, f) for f in extra_files]
    config_dict = {
        "model_name": "nougat_small",
        "handler": "handler.py",
        "version": "1.0",
        "serialized_file": None,
        "model_file": model_file,
        "extra_files": ",".join(extra_files),
        "runtime": "python",
        "export_path": pathlib.Path.home() / "model-store",
        "archive_format": "no-archive",
        "force": True,
        "requirements_file": None,
        "config_file": None,
    }
    config = ModelArchiverConfig(**config_dict)
    generate_model_archive(config)
