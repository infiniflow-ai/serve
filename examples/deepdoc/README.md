# deepdoc

This example demonstrates how to serve the [deepdoc](https://huggingface.co/InfiniFlow/deepdoc) models(det, layout.laws, layout.manual, layout, layout.paper, rec, tsr).
Refers to https://github.com/yuzhichang/ragflow/blob/fix_release/deepdoc/vision/recognizer.py.

1. docker build -t <IMAGE_NAME> .
2. docker run --rm -it -p 3000:8080 -p 3001:8081 <IMAGE_NAME> torchserve --start --model-store model_store --models all
