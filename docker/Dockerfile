# Base image with CUDA 12.1 and PyTorch support
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

WORKDIR /workspace

# Install system utilities
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and trained model weights
COPY src/ ./src/
COPY configs/ ./configs/
COPY docker/inference.py .

# Standard challenge I/O directories
# Input images are mounted into /input, output predictions written to /output
RUN mkdir -p /input /output

ENTRYPOINT ["python", "inference.py"]
