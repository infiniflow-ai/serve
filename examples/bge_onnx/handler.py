"""
Handler for BAAI embedding models: [bge-m3](https://huggingface.co/BAAI/bge-m3), [bge-large-en-v1.5](https://huggingface.co/BAAI/bge-large-en-v1.5) and [bge-large-zh-v1.5](https://huggingface.co/BAAI/bge-large-zh-v1.5).

Required python packages: torch onnxruntime-gpu transformers
"""

import os
import torch
import base64
import json
import logging
import onnxruntime as ort
from transformers import AutoTokenizer


class Bge(object):
    """
    bge-m3 handler class
    - https://github.com/FlagOpen/FlagEmbedding/issues/432#issuecomment-1948332073
    - https://huggingface.co/aapot/bge-m3-onnx
    """

    def __init__(self):
        super(Bge, self).__init__()
        self.initialized = False
        self.tokenizer = None
        self.session = None
        self.run_options = None

    def initialize(self, context):
        properties = context.system_properties
        model_dir = properties.get("model_dir")
        model_file = os.path.join(model_dir, "model.onnx")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)

        # https://github.com/microsoft/onnxruntime/issues/9509#issuecomment-951546580
        # Shrink GPU memory after execution
        self.run_options = ort.RunOptions()

        if torch.cuda.is_available() and properties.get("gpu_id") is not None:
            cuda_provider_options = {
                "device_id": properties.get("gpu_id"), # Use specific GPU
                "gpu_mem_limit": 5 * 1024 * 1024 * 1024, # Limit gpu memory
                "arena_extend_strategy": "kNextPowerOfTwo",  # gpu memory allocation strategy
            }
            session_options = ort.SessionOptions()
            self.session = ort.InferenceSession(
                model_file,
                providers=["CUDAExecutionProvider"],
                provider_options=[cuda_provider_options],
                sess_options=session_options,
            )
            gpu_id = properties.get("gpu_id")
            self.run_options.add_run_config_entry("memory.enable_memory_arena_shrinkage", f"gpu:{gpu_id}")
        else:
            self.session = ort.InferenceSession(model_file)
            self.run_options.add_run_config_entry("memory.enable_memory_arena_shrinkage", "cpu")
        self.initialized = True

    def preprocess(self, req_batch):
        assert isinstance(req_batch, list)
        req_batch_sentences = []
        req_batch_shape = []
        for req in req_batch:
            req_data = req.get("data") or req.get("body")
            print("req_data:", req_data)
            if not isinstance(req_data, list):
                req_data = json.loads(req_data)
            req_sentences = req_data
            assert isinstance(req_sentences, list)
            req_batch_shape.append(len(req_sentences))
            req_batch_sentences.extend(req_sentences)
        return req_batch_sentences, req_batch_shape

    def inference(self, data):
        req_batch_sentences, req_batch_shape = data
        inputs = self.tokenizer(req_batch_sentences, padding="longest", return_tensors="np")
        inputs_onnx = {k: ort.OrtValue.ortvalue_from_numpy(v) for k, v in inputs.items()}
        outputs = self.session.run(None, inputs_onnx, self.run_options)
        embeddings = outputs[0][:, 0, :]  # outputs[0] slice operation: (batch_size, hidden_size, 1024) -> (batch_size, 1024)
        embeddings = torch.nn.functional.normalize(torch.from_numpy(embeddings).float(), dim=-1)
        embeddings = embeddings.numpy()
        return embeddings, req_batch_shape

    def postprocess(self, data):
        embeddings, req_batch_shape = data
        resp_batch = []
        gen_idx = 0
        for i in range(len(req_batch_shape)):
            resp = embeddings[gen_idx : gen_idx + req_batch_shape[i]]
            resp = [base64.b64encode(embedding.tobytes()).decode("utf-8") for embedding in resp]
            resp_batch.append(resp)
            gen_idx += req_batch_shape[i]
        return resp_batch


_service = Bge()


def handle(data, context):
    """
    Entry point for Bge handler
    """
    try:
        if not _service.initialized:
            _service.initialize(context)

        if data is None:
            return None

        data = _service.preprocess(data)
        data = _service.inference(data)
        data = _service.postprocess(data)

        return data
    except Exception:
        logging.exception("Bge.handle got exception")
        raise
