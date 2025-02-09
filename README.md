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

Before moving on to the next step, please don't forget to set the project root path to the `CONF.PATH.BASE` in `lib/config.py`.

## Training and Evaluation

* To train the model, run the following commands:

```shell
python ScanRefer_train.py --use_multiview --use_normal --no_height --batch_size 8 --lang_num_max 32 --epoch 50 --lr 0.002 --lang_lr 0.0005 --match_lr 0.0005 --coslr
```

* To evaluate the model, run the following commands:

```shell
python ScanRefer_eval.py --folder path --reference --use_multiview --use_normal --no_height --no_nms --force --repeat 1 --lang_num_max 1 --batch_size 32
```

* To generate predictions on ScanRefer hidden test set, run the following commands:

```shell
python benchmark/predict.py --folder path --use_multiview --use_normal --no_height --batch_size 32 
```
