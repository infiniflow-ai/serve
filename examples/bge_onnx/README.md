# BAAI embedding models

This example demonstrates how to serve BAAI embedding models: [bge-m3](https://huggingface.co/BAAI/bge-m3), [bge-large-en-v1.5](https://huggingface.co/BAAI/bge-large-en-v1.5) and [bge-large-zh-v1.5](https://huggingface.co/BAAI/bge-large-zh-v1.5).

# Test

## Create mar for these models
    ```
    uv run create_mar.py BAAI/bge-m3 0
    uv run create_mar.py BAAI/bge-large-en-v1.5 0
    uv run create_mar.py BAAI/bge-large-zh-v1.5 0
    ```

## Start Serving the models

    ```
    torchserve --start --model-store ~/model-store --models all --ts-config config.properties --disable-token-auth --enable-model-api
    ```

## Run test

    ```
    uv run bge_test.py BAAI/bge-m3
    uv run bge_test.py BAAI/bge-large-en-v1.5
    uv run bge_test.py BAAI/bge-large-zh-v1.5
    ```

## Stop serving:
    ```
    torchserve --stop
    ```

## Build docker image and Run
    ```
    docker build -t infiniflow-ai/torchserve .
    docker compose up -d
    ```
