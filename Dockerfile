# DUST environment, following the README's installation steps.
# Build:  docker build -t dust .
# Run:    docker run --rm -it --gpus all --shm-size=8g -u $(id -u):$(id -g) -v "$PWD":/workspace dust
FROM python:3.12-slim

ENV PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1 HF_HOME=/workspace/.cache/huggingface

RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH

COPY requirements.txt /tmp/requirements.txt
RUN python -m pip install --upgrade pip \
    && python -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128 \
    && python -m pip install -r /tmp/requirements.txt

# Lets the container run as the host user (-u $(id -u):$(id -g)), which has no passwd entry.
ENV USER=dust HOME=/tmp
WORKDIR /workspace
CMD ["bash"]
