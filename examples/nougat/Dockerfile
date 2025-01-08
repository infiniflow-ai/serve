FROM pytorch/torchserve:latest-cpu

ARG NEED_MIRROR=0

USER root
RUN if [ "$NEED_MIRROR" == "1" ]; then \
        sed -i 's|http://archive.ubuntu.com|https://mirrors.tuna.tsinghua.edu.cn|g' /etc/apt/sources.list; \
        pip3 config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple; \
        pip3 config set global.trusted-host pypi.tuna.tsinghua.edu.cn; \
    fi

RUN pip install transformers
RUN pip install torchserve

RUN apt-get update && apt install -y zip unzip

USER model-server

RUN --mount=type=bind,source=nougat_small.mar,target=/nougat_small.mar \
    mkdir model_store && cp /nougat_small.mar model_store/

ENTRYPOINT ["torchserve", "--start", "--model-store", "model-store", "--models", "nougat_small=nougat_small.mar", "--ts-config", "config.properties", "--disable-token-auth", "--enable-model-api"]