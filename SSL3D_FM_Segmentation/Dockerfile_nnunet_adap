FROM nvidia/cuda:13.0.1-cudnn-runtime-ubuntu24.04
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
# Disable PEP 668 for Docker (safe in containers)
ENV PIP_BREAK_SYSTEM_PACKAGES=1

# Install system dependencies (Python 3.12 is default in Ubuntu 24.04)
RUN apt-get update && apt-get install -y \
    python3 \
    python3-dev \
    python3-pip \
    python3-venv \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Make sure python points to python3
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1

# Clone and install nnUNet with nnssl adaptation
WORKDIR /opt
RUN git clone https://github.com/PeterMcGor/nnUNet.git nnunet_adaptation && \
    cd nnunet_adaptation && \
    git checkout working/planning-fixes && \
    pip install -e .

# Set up nnUNet paths
ENV nnUNet_raw="/opt/nnunet_resources/nnUNet_raw"
ENV nnUNet_preprocessed="/opt/nnunet_resources/nnUNet_preprocessed"
ENV nnUNet_results="/opt/nnunet_resources/nnUNet_results"

# Create models directory and copy trained models
RUN mkdir -p /opt/nnunet_resources/models
COPY models/ /opt/nnunet_resources/models/

# Verify installation
RUN nnUNetv2_preprocess_like_nnssl --help && \
    nnUNetv2_train_pretrained --help

WORKDIR /workspace
CMD ["/bin/bash"]