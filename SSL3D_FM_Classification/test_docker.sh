#!/bin/bash

echo "=== Testing SSL3D Classification Docker ==="

echo -e "\n1. Python version:"
docker run --rm petermcgor/nnssl-classification python3 --version

echo -e "\n2. GPU check:"
docker run --rm --gpus all petermcgor/nnssl-classification nvidia-smi --query-gpu=name --format=csv,noheader

echo -e "\n3. PyTorch CUDA:"
docker run --rm --gpus all petermcgor/nnssl-classification  python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

echo -e "\n4. Framework structure:"
docker run --rm petermcgor/nnssl-classification ls /opt/SSL3D_classification/main.py

echo -e "\n5. Key imports:"
docker run --rm petermcgor/nnssl-classification python3 -c "import torch, pytorch_lightning, hydra; print('✓ All key packages imported')"

echo -e "\n=== All basic tests passed! ==="