FROM nvidia/cuda:13.0.1-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_BREAK_SYSTEM_PACKAGES=1

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Clone SSL3D classification framework
WORKDIR /opt
RUN git clone https://github.com/constantinulrich/SSL3D_classification.git
WORKDIR /opt/SSL3D_classification
RUN pip install -r requirements.txt

# Create mount points
RUN mkdir -p /workspace/tasks /workspace/models /workspace/data

# Copy pretrained model weights into the image
COPY ../models/BaseMAETrainerExtendedHealth__nnsslPlans__onemmiso /workspace/models/BaseMAETrainerExtendedHealth__nnsslPlans__onemmiso


WORKDIR /workspace
CMD ["/bin/bash"]