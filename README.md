# DeepLab V3plus
My implementation of [Deeplab_v3plus](https://arxiv.org/abs/1802.02611). This repository is based on the dataset of cityscapes and the mIOU is 70.54.

I am working with python3.5 and pytorch1.0.0 built from source. Other environments are not tested, but you need at least pytorch1.0 since I use torch.distributed to manipulate my gpus. I use two 1080ti to train my model, so you also need two gpus each of which should have at least 9G memory.


## Dataset
The experiment is done with the dataset of [CityScapes](https://www.cityscapes-dataset.com/). You need to register on the website and download the dataset images and annotations. Then you create a `data` directory and then decompress.
```
    $ cd DeepLabv3plus
    $ mkdir -p data
    $ mv /path/to/leftImg8bit_trainvaltest.zip data
    $ mv /path/to/gtFine_trainvaltest.zip data
    $ cd data
    $ unzip leftImg8bit_trainvaltest.zip
    $ unzip gtFine_trainvaltest.zip
```


## Train && Eval
After creating the dataset, you can train on the cityscapes train set and evaluate on the validation set.  
Train: 
```
    $ cd DeepLabv3plus
    $ CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train.py
```
This will take around 13 hours on two 1080ti gpus. After training, the model will be evaluated on the val set automatically, and you will see a mIOU of 70.54.

Eval:
If you want to evaluate a trained model, you can also do this: 
```
    $ python evaluate.py
```
or if you want to evaluate on multi-gpus, you can also do this: 
```
    $ CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 evaluate.py
```


## Configurations
* If you want to use your own dataset, you may implement you `dataset` file as does with my `cityscapes.py`. 

* As for the hyper-parameters, you may change them in the configuration file `configs/configurations.py`.


## Pretrained Model
If you need model parameters pretrained on cityscapes, you can download the `pth` file [here](https://pan.baidu.com/s/1vbFxwchQybi77drB6divww) with extraction code: `3i4g`.
