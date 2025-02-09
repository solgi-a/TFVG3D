# TFVG3D: A Transformer-based Framework for Visual Grounding on 3D Point Clouds

This repo is the official implementation of "[A Transformer-based Framework for Visual Grounding on 3D Point Clouds](https://ieeexplore.ieee.org/document/10475280)".

![Diagram](./image/Diagram.jpg)

Please refer to the paper for more details and visit the [ScanRefer](https://github.com/daveredrum/ScanRefer) repository for instructions on downloading the dataset.

## Necessary Packages 
All experiments were conducted using a single RTX 4090-12GB GPU.
* CUDA: 11.8
* PyTorch: 2.4.0
* python: 3.12

Execute the following command to install PyTorch:
```shell
pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu118
```

Install the necessary packages listed out in `req/requirements.txt`:
```shell
pip install -r req/requirements.txt
```

Run the following commands to compile the CUDA modules for the PointNet++ backbone:
```shell
cd pointnet2_ops_lib
python setup.py install
```
