# Cloud Detection on WHUS2‑CD+ 
- Bachelor thesis of Computer Science Engineering at Sapienza University of Rome, academic year 2022/2023 and 
- Project for the *Artificial Intelligence Laboratory* course

[comment]: <> (TODO: image with main results)

## Table of contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Technologies](#technologies)
- [Acknowledgements](#acknowledgements)

## Introduction

The Sentinel-2A satellite provides high resolution multispectral images of the Earth's surface in the visible and near-infrared domain with a spatial resolution of 10m, 20m and 60m. However, these images may be affected by clouds and cloud shadows, which can cover whole areas and obstaculate the analysis of the images. It is therefore necessary to develop a cloud detection algorithm to identify the presence of clouds with high accuracy, which is the aim of this project.

In particular, one of the possibile applications of such algorithm consists in the detection of clouds with the goal of masking the corresponding pixels during the creation of a composite of images taken in different dates. 
This process allows to obtain images of the Earth's surface without clouds, or lessen their presence.

[comment]: <> (collassare Dataset e Technologies in un'unica sezione source?)
## Dataset
- [WHUS2-CD+](https://zenodo.org/record/5511793#.ZGIjmdJBzHW), cloud validation detection dataset for Sentinel-2A images over China

## Technologies
Created with:
- [Python 3.11](https://www.python.org/downloads/release/python-3111/)
- [PyTorch 2.0.1](https://pytorch.org/get-started/pytorch-2.0/)
- U-Net convolutional autoencoder as machine learning architecture 

## Acknowledgements
Paper on which the project is based:
  - [\[arXiv:2105.00967\]](https://arxiv.org/abs/2105.00967v1) A lightweight deep learning based cloud detection method for Sentinel-2A imagery fusing multi-scale spectral and spatial features
    - Jun Li, Zhaocong Wu, Zhongwen Hu, Canliang Jian, Shaojie Luo, Lichao Mou, Xiao Xiang Zhu, Matthieu Molinier

    
