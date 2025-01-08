"""
Handler for BAAI embedding models: [bge-m3](https://huggingface.co/BAAI/bge-m3), [bge-large-en-v1.5](https://huggingface.co/BAAI/bge-large-en-v1.5) and [bge-large-zh-v1.5](https://huggingface.co/BAAI/bge-large-zh-v1.5).

Required python packages: torch vllm

BGE models give following error if the input is too long:

2025-02-06T12:01:13,594 [INFO ] W-9002-bge-large-zh-v1.5_1.0-stdout MODEL_LOG - WARNING 02-06 12:01:13 scheduler.py:944] Input prompt (612 tokens) is too long and exceeds limit of 512
2025-02-06T12:01:13,641 [INFO ] W-9002-bge-large-zh-v1.5_1.0-stdout MODEL_LOG - Bge.handle got exception
2025-02-06T12:01:13,641 [INFO ] W-9002-bge-large-zh-v1.5_1.0-stdout MODEL_LOG - Traceback (most recent call last):
2025-02-06T12:01:13,642 [INFO ] W-9002-bge-large-zh-v1.5_1.0-stdout MODEL_LOG -   File "/home/model-server/model-store/bge-large-zh-v1.5/handler.py", line 83, in handle
2025-02-06T12:01:13,642 [INFO ] W-9002-bge-large-zh-v1.5_1.0-stdout MODEL_LOG -     data = _service.inference(data)
2025-02-06T12:01:13,642 [INFO ] W-9002-bge-large-zh-v1.5_1.0-stdout MODEL_LOG -   File "/home/model-server/model-store/bge-large-zh-v1.5/handler.py", line 51, in inference
2025-02-06T12:01:13,642 [INFO ] W-9002-bge-large-zh-v1.5_1.0-stdout MODEL_LOG -     outputs = self.llm.encode(req_batch_sentences)
2025-02-06T12:01:13,642 [INFO ] W-9002-bge-large-zh-v1.5_1.0-stdout MODEL_LOG -   File "/home/venv/lib/python3.9/site-packages/vllm/utils.py", line 1021, in inner
2025-02-06T12:01:13,642 [INFO ] W-9002-bge-large-zh-v1.5_1.0-stdout MODEL_LOG -     return fn(*args, **kwargs)
2025-02-06T12:01:13,642 [INFO ] W-9002-bge-large-zh-v1.5_1.0-stdout MODEL_LOG -   File "/home/venv/lib/python3.9/site-packages/vllm/entrypoints/llm.py", line 871, in encode
2025-02-06T12:01:13,642 [INFO ] W-9002-bge-large-zh-v1.5_1.0-stdout MODEL_LOG -     outputs = self._run_engine(use_tqdm=use_tqdm)
2025-02-06T12:01:13,642 [INFO ] W-9002-bge-large-zh-v1.5_1.0-stdout MODEL_LOG -   File "/home/venv/lib/python3.9/site-packages/vllm/entrypoints/llm.py", line 1242, in _run_engine
2025-02-06T12:01:13,642 [INFO ] W-9002-bge-large-zh-v1.5_1.0-stdout MODEL_LOG -     step_outputs = self.llm_engine.step()
2025-02-06T12:01:13,642 [INFO ] W-9002-bge-large-zh-v1.5_1.0-stdout MODEL_LOG -   File "/home/venv/lib/python3.9/site-packages/vllm/engine/llm_engine.py", line 1439, in step
2025-02-06T12:01:13,642 [INFO ] W-9002-bge-large-zh-v1.5_1.0-stdout MODEL_LOG -     self._process_model_outputs(ctx=ctx)
2025-02-06T12:01:13,642 [INFO ] W-9002-bge-large-zh-v1.5_1.0-stdout MODEL_LOG -   File "/home/venv/lib/python3.9/site-packages/vllm/engine/llm_engine.py", line 1190, in _process_model_outputs
2025-02-06T12:01:13,642 [INFO ] W-9002-bge-large-zh-v1.5_1.0-stdout MODEL_LOG -     request_output = RequestOutputFactory.create(
2025-02-06T12:01:13,643 [INFO ] W-9002-bge-large-zh-v1.5_1.0-stdout MODEL_LOG -   File "/home/venv/lib/python3.9/site-packages/vllm/outputs.py", line 391, in create
2025-02-06T12:01:13,643 [INFO ] W-9002-bge-large-zh-v1.5_1.0-stdout MODEL_LOG -     return RequestOutput.from_seq_group(seq_group, use_cache,
2025-02-06T12:01:13,643 [INFO ] W-9002-bge-large-zh-v1.5_1.0-stdout MODEL_LOG -   File "/home/venv/lib/python3.9/site-packages/vllm/outputs.py", line 185, in from_seq_group
2025-02-06T12:01:13,643 [INFO ] W-9002-bge-large-zh-v1.5_1.0-stdout MODEL_LOG -     raise ValueError(
2025-02-06T12:01:13,643 [INFO ] W-9002-bge-large-zh-v1.5_1.0-stdout MODEL_LOG - ValueError: Sampling parameters are missing for a CompletionRequest.


https://github.com/vllm-project/vllm/pull/4598 add truncate_prompt_tokens to work offline, but it has been closed for unknown reason.

I have to truncate manually at preprocess.
"""

import base64
import json
import logging
import os
import pynvml

class Bge(object):
    """
    bge-m3 handler class
    - https://github.com/FlagOpen/FlagEmbedding/issues/1060
    - https://github.com/FlagOpen/FlagEmbedding/issues/987
    """

    MAX_TOKENS = {"bge-m3": 8000, "bge-large-en-v1.5": 500, "bge-large-zh-v1.5": 500}
    def __init__(self):
        super(Bge, self).__init__()
        self.initialized = False
        self.llm = None

    def initialize(self, context):
        properties = context.system_properties
        model_dir = properties.get("model_dir")
        model_name = properties.get("model_name")
        self.max_tokens = self.MAX_TOKENS.get(model_name, 500)
        gpu_id = properties.get("gpu_id")
        print(f"gpu_id: {gpu_id}")
        if gpu_id is not None:
            # This env shall be populated BEFORE CUDA initailization(importing torch or vllm does)
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            # Check if only the given GPU is available
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(int(gpu_id))
            gpu_name = pynvml.nvmlDeviceGetName(handle)
            gpu_uuid = pynvml.nvmlDeviceGetUUID(handle)
            print(f"GPU {gpu_id}: {gpu_name} ({gpu_uuid})")
            pynvml.nvmlShutdown()
            import torch
            assert torch.cuda.is_available()==True
            assert torch.cuda.device_count()==1
            assert torch.cuda.get_device_name(0)==gpu_name
        import vllm
        self.llm = vllm.LLM(model_dir)
        self.tokenizer = self.llm.get_tokenizer()
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
            for i in range(len(req_sentences)):
                ids = self.tokenizer.encode(req_sentences[i])
                if len(ids) > self.max_tokens:
                    print(f'before truncation({len(ids)} tokens): {req_sentences[i]}')
                    req_sentences[i] = self.tokenizer.decode(ids[:self.max_tokens], skip_special_tokens=True)
                    print(f'after truncation({self.max_tokens} tokens): {req_sentences[i]}')
            req_batch_shape.append(len(req_sentences))
            req_batch_sentences.extend(req_sentences)
        return req_batch_sentences, req_batch_shape

    def inference(self, data):
        req_batch_sentences, req_batch_shape = data
        outputs = self.llm.encode(req_batch_sentences)
        print(f"outputs: {outputs}")
        embeddings = [output.outputs.data.numpy() for output in outputs]
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
