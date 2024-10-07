# Python version can be changed, e.g.
# FROM python:3.8
# FROM ghcr.io/mamba-org/micromamba:1.5.1-focal-cuda-11.3.1
FROM docker.io/python:3.12.1-slim-bookworm

LABEL org.opencontainers.image.authors="FNNDSC <dev@babyMRI.org>" \
      org.opencontainers.image.title="pl-uae_gen" \
      org.opencontainers.image.description="Generate UAE values from the TF images"

ARG SRCDIR=/usr/local/src/pl-uae_gen
WORKDIR ${SRCDIR}

COPY requirements.txt .
RUN apt-get update && \
    apt-get install -y \
        git \
        python3-pip \
        python3-dev \
        python3-opencv \
        libglib2.0-0
RUN python3 -m pip install --upgrade pip
RUN pip3 install torch -f https://download.pytorch.org/whl/cu111/torch_stable.html
RUN pip install -r requirements.txt

COPY . .
ARG extras_require=none
RUN pip install ".[${extras_require}]" \
    && cd / && rm -rf ${SRCDIR}
WORKDIR /

CMD ["uae_gen"]
