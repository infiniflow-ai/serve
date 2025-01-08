"""
Handler for [nougat-small](facebook/nougat-small), a model for PDF-to-markdown.
Required python packages: pillow torch transformers levenshtein nltk
"""

import os
import io
import base64
import json
from PIL import Image
import torch
from transformers import AutoProcessor, VisionEncoderDecoderModel

from transformers import StoppingCriteria, StoppingCriteriaList
from collections import defaultdict


class RunningVarTorch:
    def __init__(self, L=15, norm=False):
        self.values = None
        self.L = L
        self.norm = norm

    def push(self, x: torch.Tensor):
        assert x.dim() == 1
        if self.values is None:
            self.values = x[:, None]
        elif self.values.shape[1] < self.L:
            self.values = torch.cat((self.values, x[:, None]), 1)
        else:
            self.values = torch.cat((self.values[:, 1:], x[:, None]), 1)

    def variance(self):
        if self.values is None:
            return
        if self.norm:
            return torch.var(self.values, 1) / self.values.shape[1]
        else:
            return torch.var(self.values, 1)


class StoppingCriteriaScores(StoppingCriteria):
    def __init__(self, threshold: float = 0.015, window_size: int = 200):
        super().__init__()
        self.threshold = threshold
        self.vars = RunningVarTorch(norm=True)
        self.varvars = RunningVarTorch(L=window_size)
        self.stop_inds = defaultdict(int)
        self.stopped = defaultdict(bool)
        self.size = 0
        self.window_size = window_size

    @torch.no_grad()
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        last_scores = scores[-1]
        self.vars.push(last_scores.max(1)[0].float().cpu())
        self.varvars.push(self.vars.variance())
        self.size += 1
        if self.size < self.window_size:
            return False

        varvar = self.varvars.variance()
        for b in range(len(last_scores)):
            if varvar[b] < self.threshold:
                if self.stop_inds[b] > 0 and not self.stopped[b]:
                    self.stopped[b] = self.stop_inds[b] >= self.size
                else:
                    self.stop_inds[b] = int(
                        min(max(self.size, 1) * 1.15 + 150 + self.window_size, 4095)
                    )
            else:
                self.stop_inds[b] = 0
                self.stopped[b] = False
        return all(self.stopped.values()) and len(self.stopped) > 0


class Nougat(object):
    """
    Nougat handler class. Nougat model trained on PDF-to-markdown.
    Ref - https://huggingface.co/facebook/nougat-small
    """

    def __init__(self):
        super(Nougat, self).__init__()
        self.initialized = False
        self.processor = None
        self.model = None
        self.device = None

    def initialize(self, context):
        self.manifest = context.manifest
        properties = context.system_properties
        model_dir = properties.get("model_dir")
        model_file = self.manifest["model"]["modelFile"]
        model_path = os.path.join(model_dir, model_file)
        if not os.path.isfile(model_path):
            raise RuntimeError(f"Missing the model file {model_path}")

        self.processor = AutoProcessor.from_pretrained(model_dir)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_dir)
        self.device = torch.device(
            "cuda:" + str(properties.get("gpu_id"))
            if torch.cuda.is_available()
            else "cpu"
        )
        self.model.to(self.device)
        self.initialized = True

    def preprocess(self, req_batch):
        assert isinstance(req_batch, list)
        req_batch_images = []
        req_batch_shape = []
        for req in req_batch:
            req_data = req.get("data") or req.get("body")
            # print("req_data:", req_data)
            if not isinstance(req_data, dict):
                req_data = json.loads(req_data)
            req_images = req_data["images"]
            req_batch_shape.append(len(req_images))
            req_images = [
                base64.b64decode(image.encode("utf-8")) for image in req_images
            ]
            req_images = [Image.open(io.BytesIO(image)) for image in req_images]
            req_batch_images.extend(req_images)
        return req_batch_images, req_batch_shape

    def inference(self, data):
        # autoregressively generate tokens, with custom stopping criteria (as defined by the Nougat authors)
        # prepare image for the model
        req_batch_images, req_batch_shape = data
        pixel_values = self.processor(
            images=req_batch_images, return_tensors="pt"
        ).pixel_values
        # print("pixel_values.shape", pixel_values.shape)
        outputs = self.model.generate(
            pixel_values.to(self.device),
            min_length=1,
            max_length=3584,
            bad_words_ids=[[self.processor.tokenizer.unk_token_id]],
            return_dict_in_generate=True,
            output_scores=True,
            stopping_criteria=StoppingCriteriaList([StoppingCriteriaScores()]),
        )
        generated = self.processor.batch_decode(outputs[0], skip_special_tokens=True)
        generated = self.processor.post_process_generation(
            generated, fix_markdown=False
        )
        return generated, req_batch_shape

    def postprocess(self, data):
        generated, req_batch_shape = data
        resp_batch = []
        gen_idx = 0
        for i in range(len(req_batch_shape)):
            resp_batch.append(generated[gen_idx : gen_idx + req_batch_shape[i]])
            gen_idx += req_batch_shape[i]
        return resp_batch


_service = Nougat()


def handle(data, context):
    """
    Entry point for Nougat handler
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
    except Exception as e:
        raise Exception("Unable to process input data. " + str(e))
