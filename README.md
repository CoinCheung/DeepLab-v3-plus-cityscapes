# DeepLab V3plus
My implementation of [Deeplab_v3plus](https://arxiv.org/abs/1802.02611). 

This project is based on python3.5 and pytorch1.0.0 built from source.


## Dataset
The experiment is done with the dataset of [CityScapes](https://www.cityscapes-dataset.com/). You need to regiester on the website and download the dataset images and annotations. Then you create a `data` directory and then decompress.
```
    $ cd Deeplab_v3_plus
    $ mkdir -p data
    $ mv /path/to/leftImg8bit_trainvaltest.zip data
    $ mv /path/to/gtFine_trainvaltest.zip data
    $ cd data
    $ unzip leftImg8bit_trainvaltest.zip
    $ unzip gtFine_trainvaltest.zip
```


## Train && Eval
After creating the dataset, you can train on the cityscapes train set and evaluate on the test set.
Train: 
```
    $ cd Deeplab_v3_plus
    $ python train.py
```

Eval:
```
    $ python evaluate.py
```


## Notes:
* The original paper have no information of the cropsize on cityscapes. Considering the limit of memory usage, I employ the crop size of `(320, 240)`. This may cause some deterioration of the accuracy.
