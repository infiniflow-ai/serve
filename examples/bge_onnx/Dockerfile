FROM pytorch/torchserve:latest-gpu

# https://github.com/pytorch/serve/issues/3374
# FROM fabridamicelli/torchserve:latest-gpu-python3.10

ARG NEED_MIRROR=0
ENV DEBIAN_FRONTEND=noninteractive

USER root
SHELL ["/bin/bash", "-c"]

RUN --mount=type=cache,id=ragflow_uv,target=/root/.cache/uv,sharing=locked \
    if [ "$NEED_MIRROR" == "1" ]; then \
        sed -i 's|http://archive.ubuntu.com|https://mirrors.tuna.tsinghua.edu.cn|g' /etc/apt/sources.list; \
        pip3 config set global.index-url https://mirrors.pku.edu.cn/pypi/web/simple; \
        pip3 config set global.trusted-host mirrors.pku.edu.cn; \
        mkdir -p /etc/uv && \
        echo "[[index]]" > /etc/uv/uv.toml && \
        echo 'url = "https://pypi.tuna.tsinghua.edu.cn/simple"' >> /etc/uv/uv.toml && \
        echo "default = true" >> /etc/uv/uv.toml; \
    fi && \
    pip3 install uv && \
    uv pip install -U transformers onnxruntime-gpu numpy pillow levenshtein nltk

COPY config.properties ./

# docker/Dockerfile has initialized ENTRYPOINT and CMD
