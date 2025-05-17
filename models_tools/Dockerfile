FROM nvidia/cuda:12.2.8-cudnn8-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y \
    python3-pip python3-dev git curl && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir jupyterlab

RUN pip3 install --no-cache-dir tensorflow[and-cuda]

WORKDIR /workspace
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]