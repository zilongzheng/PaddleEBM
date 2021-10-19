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
- Patchwise Generative ConvNet: Training Energy-Based Models from a Single Natural Image for Internal Learning. CVPR 2021 [[Pdf](https://openaccess.thecvf.com/content/CVPR2021/papers/Zheng_Patchwise_Generative_ConvNet_Training_Energy-Based_Models_From_a_Single_Natural_CVPR_2021_paper.pdf)][[Model](https://github.com/zilongzheng/PaddleEBM/blob/dev/models/patchgencn_model.py)][[Config](https://github.com/zilongzheng/PaddleEBM/blob/dev/configs/patchgencn_single.yaml)]
- Learning Cycle-Consistent Cooperative Networks via Alternating MCMC Teaching for Unsupervised Cross-Domain Translation. AAAI 2021
- Generative VoxelNet: Learning Energy-Based Models for 3D Shape Synthesis and Analysis. TPAMI 2020 [[Pdf](https://arxiv.org/pdf/2012.13522.pdf)]
- Cooperative Training of Descriptor and Generator Networks. TPAMI 2018 [[Pdf](https://arxiv.org/pdf/1609.09408.pdf)][[Model](./models/coopnets_model.py)]


## Acknowledgements
This repository is based on [PaddleGAN](https://github.com/PaddlePaddle/PaddleGAN) and [PaddlePaddle](https://github.com/PaddlePaddle/Paddle) official implementation.