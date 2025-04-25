# FROM ubuntu:22.04
FROM python:3.9

ENV DEBIAN_FRONTEND=noninteractive
# ENV TZ=UTC

# Install basic apt packages.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    nano \
    vim \
    wget \
    curl

RUN pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124

RUN pip install \
    mmengine-lite==0.10.4 scikit-learn==1.5.0 tqdm==4.66.5 \
    pyarrow==16.1.0 numpy==1.26.4 matplotlib==3.9.0 pandas==2.2.2
RUN pip install requests==2.32.3
RUN pip install optuna scipy sympy einops
RUN pip install timm==1.0.9
# RUN pip install lightgbm
RUN pip install packaging
RUN pip install netCDF4 xarray cartopy h5py bottleneck
RUN pip install compressai tensorboard

ENTRYPOINT /bin/bash
