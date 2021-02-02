# ASEGAN: Speech Enhancement Generative Adversarial Network Based on Asymmetric AutoEncoder
[中文版简介](README.md)  
[Readme with English Version](README_en.md)

#### Introduction
An variant of [SEGAN](https://arxiv.org/abs/1703.09452)  
the original full convolution structure is replaced by the self-designed asymmetric autoencoder structure, which makes the model lighter while maintaining the original performance(maybe better)

#### Net Architecture
![structure](doc/structure.png)

#### Installation

1.  Install necessary Libraries  
    [Anaconda](https://www.anaconda.com/)  
    [cuda](https://developer.nvidia.com/zh-cn/cuda-toolkit)  
    [cudnn](https://developer.nvidia.com/zh-cn/cudnn)
2.  Create an env of python  
    `conda create -n ASEGAN python=3.8`  
    `conda activate ASEGAN`
3.  Install package   
    `pip install -r requirements.txt`  
    or`conda install --yes --file requirements.txt`

#### Instructions
    Check the file 'config/config.yaml' before use it  
    The data folder structure refers to the structure in 'data/'
1.  Data Pre-process
    `python data_preprocess.py`
2.  Train a model  
    `python train.py`
3.  Eval a model  
    `python test.py`
    
#### Pre Training Model Download
[百度网盘，提取码6793](https://pan.baidu.com/s/11xzTzrP7WkchQWk55Z0bKw)  
[GoogleDrive](https://drive.google.com/drive/folders/1RVKEbCnQyEMmA6JOoqSNnbRnLEghdhvq?usp=sharing)  
file name rule is ‘dataset-epoch-numofdata.pkl’