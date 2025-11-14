# CSRN-INR: An Enhanced Continuous Super-Resolution Method for Remote Sensing Images Based on Implicit Neural Representations

# Environment
Our code is based on PyTorch 1.12.1 and Python 3.8.20. Training is performed on an NVIDIA RTX 2080 Ti GPU.

# Train
Before training, revise the path in config file according to your setting：configs/cmsr/
## 1. Download Training Dataset

## 2. Train Your Model
  ```bash
  python train.py --config configs/cmsr/init-div2k-x2.yaml --gpu 0
  ```
