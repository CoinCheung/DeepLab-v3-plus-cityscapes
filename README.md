# DeepLab V3plus
My implementation of [Deeplab_v3plus](https://arxiv.org/abs/1802.02611). This repository is based on the dataset of cityscapes and the mIOU is 80.12.

I am working with python3.5 and pytorch1.0.0 built from source. Other environments are not tested, but you need at least pytorch1.0 since I use torch.distributed to manipulate my gpus. I use two 1080ti to train my model, so you also need two gpus each of which should have at least 9G memory.


## Dataset
The experiment is done with the dataset of [CityScapes](https://www.cityscapes-dataset.com/). You need to register on the website and download the dataset images and annotations. Then you create a `data` directory and then decompress.
```
    $ cd DeepLab-v3-plus
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
    $ cd Deeplab_v3_plus
    $ CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train.py
```
This will take around 13 hours on two 1080ti gpus. After training, the model will be evaluated on the val set automatically, and you will see a mIOU of 80.12.

Eval:
If you want to evaluate a trained model, you can also do this: 
```
    $ python evaluate.py
```

