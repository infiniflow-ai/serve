"""
Handler for [deepdoc](InfiniFlow/deepdoc) models

Required python packages: torch numpy onnxruntime-gpu
"""

import os
import glob
import numpy as np
import base64
import json
import logging
import pynvml


class Deepdoc(object):
    """
    Deepdoc handler class. yolo model trained on image-to-boxes.
        - https://github.com/ultralytics/ultralytics/blob/main/docs/en/models/yolov10.md
        - https://huggingface.co/jameslahm/yolov10x
        - https://huggingface.co/onnx-community/YOLOv10
    """

    def __init__(self):
        super(Deepdoc, self).__init__()
        self.initialized = False
        self.processor = None
        self.sessions = {}  # model_name -> ort.InferenceSession

    def initialize(self, context):
        properties = context.system_properties
        model_dir = properties.get("model_dir")
        model_files = sorted(
            glob.glob(os.path.join(model_dir, "*.onnx"), recursive=False)
        )

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

        import onnxruntime as ort
        # https://github.com/microsoft/onnxruntime/issues/9509#issuecomment-951546580
        # Shrink GPU memory after execution
        self.run_options = ort.RunOptions()
        if gpu_id is not None:
            gpu_id = properties.get("gpu_id")
            self.run_options.add_run_config_entry("memory.enable_memory_arena_shrinkage", f"gpu:{gpu_id}")
        else:
            self.run_options.add_run_config_entry("memory.enable_memory_arena_shrinkage", "cpu")

        for model_file in model_files:
            model_name = os.path.splitext(os.path.basename(model_file))[0]
            if gpu_id is not None:
                cuda_provider_options = {
                    "device_id": gpu_id, # Use specific GPU
                    "gpu_mem_limit": 1 * 1024 * 1024 * 1024, # Limit gpu memory
                    "arena_extend_strategy": "kNextPowerOfTwo",  # gpu memory allocation strategy
                }
                session_options = ort.SessionOptions()
                session = ort.InferenceSession(
                    model_file,
                    providers=["CUDAExecutionProvider"],
                    provider_options=[cuda_provider_options],
                    sess_options=session_options,
                )
            else:
                session = ort.InferenceSession(model_file)
            self.sessions[model_name] = session
        self.initialized = True

    def preprocess(self, req_batch):
        assert isinstance(req_batch, list)
        req_batch_images = []
        req_batch_shape = []
        req_batch_model = []
        for req in req_batch:
            req_data = req.get("data") or req.get("body")
            # print("req_data:", req_data)
            if not isinstance(req_data, dict):
                req_data = json.loads(req_data)
            req_model = req_data["model"]
            req_images = req_data["images"]
            for i in range(len(req_images)):
                req_batch_model.append(req_model)
            req_batch_shape.append(len(req_images))
            req_images = [
                base64.b64decode(image.encode("utf-8")) for image in req_images
            ]
            req_images = [np.frombuffer(image, dtype=np.uint8).reshape(1, 3, 1024, 1024) for image in req_images]
            req_batch_images.extend(req_images)
        return req_batch_images, req_batch_model, req_batch_shape

    def inference(self, data):
        resp_batch_boxes = []
        req_batch_images, req_batch_model, req_batch_shape = data
        req_batch_images = [image.astype(np.float32) for image in req_batch_images]
        for i in range(len(req_batch_images)):
            image = req_batch_images[i]
            # print("image.shape", image.shape) # (1, 3, 640, 640)
            session = self.sessions[req_batch_model[i]]
            results = session.run(None, {"images": image}, self.run_options)
            boxes = results[0]
            # print("boxes", boxes) # (1, x, 6)
            reshaped_boxes = boxes.reshape(boxes.shape[1], boxes.shape[2])  # 直接调整为 (x, 6)
            resp_batch_boxes.append(reshaped_boxes)
        return resp_batch_boxes, req_batch_shape

    def postprocess(self, data):
        resp_batch_boxes, req_batch_shape = data
        resp_batch = []
        gen_idx = 0
        for i in range(len(req_batch_shape)):
            resp = resp_batch_boxes[gen_idx : gen_idx + req_batch_shape[i]]
            resp = [base64.b64encode(boxes.tobytes()).decode("utf-8") for boxes in resp]
            resp_batch.append(resp)
            gen_idx += req_batch_shape[i]
        return resp_batch


_service = Deepdoc()


def handle(data, context):
    """
    Entry point for Deepdoc handler
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
        logging.exception("deepdoc.handle got exception")
        raise
