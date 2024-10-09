# Python version can be changed, e.g.
# FROM python:3.8
# FROM ghcr.io/mamba-org/micromamba:1.5.1-focal-cuda-11.3.1
FROM pytorch/pytorch:2.4.1-cuda11.8-cudnn9-runtime
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=America/New_York 


LABEL org.opencontainers.image.authors="FNNDSC <dev@babyMRI.org>" \
      org.opencontainers.image.title="pl-uae_gen" \
      org.opencontainers.image.description="Generate UAE values from the TF images"

ARG SRCDIR=/usr/local/src/pl-uae_gen
WORKDIR ${SRCDIR}

COPY requirements.txt .
RUN python3 -m pip install --upgrade pip
RUN pip install -r requirements.txt

COPY . .
ARG extras_require=none
RUN pip install ".[${extras_require}]" \
    && cd / && rm -rf ${SRCDIR}
    
WORKDIR /

CMD ["uae_gen"]
