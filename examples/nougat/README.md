# nougat

This example demonstrates how to use the `torchserve` to serve a HuggingFace model.
Refers to https://github.com/pytorch/serve/issues/390.

The [Nougat](https://huggingface.co/docs/transformers/en/model_doc/nougat) model uses the same architecture as Donut, meaning an image Transformer encoder and an autoregressive text Transformer decoder to translate scientific PDFs to text.

1. docker build -t <IMAGE_NAME> .
2. docker run --rm -it -p 3000:8080 -p 3001:8081 <IMAGE_NAME> torchserve --start --model-store model_store --models all
