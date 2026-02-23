FROM pytorch/pytorch:2.2.0-cuda11.8-cudnn8-runtime

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    vim \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir /configfiles
COPY requirements.txt /configfiles/
RUN pip install --no-cache-dir -r /configfiles/requirements.txt

WORKDIR /app
COPY . /app
ENV PYTHONPATH="/app/src:/app/GWASmt/src:${PYTHONPATH}"