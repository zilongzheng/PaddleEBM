# PaddleEBM

This repository is a PaddlePaddle Library of Energy-based Models (EBMs). 

## Prerequisites
- Linux or macOS
- Python 3.6+
- PaddlePaddle 2.0+

## Getting Started
- Clone this repo:
```
git clone https://github.com/zilongzheng/PaddleEBM.git
cd PaddleEBM
```
- Install Python dependencies by
```
pip install -r requirements.txt
```

### Train
- Train a model (e.g. CoopNets on MNIST dataset):
```
python train.py --config-file configs/coopnets_mnist.yaml
```

## Publication
- Learning Energy-Based Model with Variational Auto-Encoder as Amortized Sampler. AAAI 2021 [[Pdf](https://arxiv.org/pdf/2012.14936.pdf)]
- Learning Cycle-Consistent Cooperative Networks via Alternating MCMC Teaching for Unsupervised Cross-Domain Translation. AAAI 2021
- Generative VoxelNet: Learning Energy-Based Models for 3D Shape Synthesis and Analysis. TPAMI 2020 [[Pdf](https://arxiv.org/pdf/2012.13522.pdf)]
- Cooperative Training of Descriptor and Generator Networks. TPAMI 2018 [[Pdf](https://arxiv.org/pdf/1609.09408.pdf)][[Model](./models/coopnets_model.py)]


## Acknowledgements
This repository is based on [PaddleGAN](https://github.com/PaddlePaddle/PaddleGAN) and [PaddlePaddle](https://github.com/PaddlePaddle/Paddle) official implementation.