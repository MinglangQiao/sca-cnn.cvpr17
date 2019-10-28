# SCA-CNN

> Source code for the paper: [SCA-CNN: Spatial and Channel-wise Attention in Convolution Networks for Imgae Captioning](https://arxiv.org/abs/1611.05594)

This code is based on [arctic-captions](https://github.com/kelvinxu/arctic-captions) and [arctic-capgen-vid](https://github.com/yaoli/arctic-capgen-vid).

This code is only for two-layered attention model in ResNet-152 Network for MS COCO dataset. Other networks (VGG-19) or datasets (Flickr30k/Flickr8k) can also be used with minor modifications.

## Dependencies
* A python library: [Theano](http://www.deeplearning.net/software/theano/).

* Other python package dependencies like **numpy/scipy, skimage, opencv, sklearn, hdf5** which can be installed by `pip`, or simply run  
  ~~~
  $ pip install -r requirements.txt
  ~~~

* [Caffe](http://caffe.berkeleyvision.org/) for image CNN feature extraction. You should install caffe and building the pycaffe interface to extract the image CNN feature. 

* The official coco evaluation scrpits [coco-caption](https://github.com/tylin/coco-caption) for results evaluation. Install it by simply adding it into `$PYTHONPATH`.

## Getting Started
1. **Get the code** `$ git clone` the repo and install the dependencies

2. **Save the pretrained CNN weights** Save the ResNet-152 weights pretrained on ImageNet. Before running the code, set the variable *deploy* and *model* in *save_resnet_weight.py* to your own path. Then run:
  ~~~
  $ cd cnn
  $ python save_resnet_weight.py
  ~~~
3. **Preprocessing the dataset** For the preprocessing of captioning, we directly use the processed JSON blob from [neuraltalk](http://cs.stanford.edu/people/karpathy/deepimagesent/). Similar to step 2, set the `PATH` in *cnn_until.py* and *make_coco.py* to your own install path. Then run:
  ~~~
  $ cd data
  $ python make_coco.py # rename the raw  layer152 folder to generate a new one before run this
  ~~~
4. **Training**  The results are saved in the directory `exp`.
  ~~~
  $ THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python sca_resnet_branch2b.py
  ~~~

## Citation

If you find this code useful, please cite the following paper:

  ```
  @inproceedings{chen2016sca,
    title={SCA-CNN: Spatial and Channel-wise Attention in Convolutional Networks for Image Captioning},
    author={Chen, Long and Zhang, Hanwang and Xiao, Jun and Nie, Liqiang and Shao, Jian and Liu, Wei and Chua, Tat-Seng},
    booktitle={CVPR},
    year={2017}
  }
  ```

### ml add
env build
```
conda create -n sca python=2.7

pip install Theano
pip install -r requirements.txt

```
using the sca python2.7 env


查看文件数
```
ls -l|grep "^-"| wc -l

```

build python2.7 version caffe
```
首先复制一份之前配好的python3的caffe-master-v3_2， 重命名为caffe-master_py27

然后
conda create -n py27 python=2.7

然后修改makefile.config，启用
PYTHON_INCLUDE := /usr/include/python2.7 \
 		/usr/lib/python2.7/dist-packages/numpy/core/include

ANACONDA_HOME := $(HOME)/anaconda3/envs/py27

PYTHON_INCLUDE := $(ANACONDA_HOME)/include \
		 $(ANACONDA_HOME)/include/python2.7 \
		 $(ANACONDA_HOME)/lib/python2.7/site-packages/numpy/core/include

注释掉这个
# PYTHON_LIBRARIES := boost_python-py36 python3.6m

```

编译后import时会报这个错
```
from numpy.lib.arraypad import _validate_lengths
ImportError: cannot import name _validate_lengths
```
然后
```
conda install numpy==1.14.5
```
然后又会报这个错
```
from google.protobuf.internal import enum_type_wrapper
ImportError: No module named google.protobuf.internal
```
这么解决
```
pip install protobuf
```
Bingo!


如果因为这个(不要运行下面这个命令)
```
pip install numpy==1.15.0
```
报这个错
```
from ._caffe import Net, SGDSolver, NesterovSolver, AdaGradSolver, \
ImportError: numpy.core.multiarray failed to import
```
参考[这里](https://blog.csdn.net/qq_32458499/article/details/82849577)解决，然后
```
import numpy as np
ImportError: No module named numpy
```
运行
```
cd ~/1_caption_saliency_ref_code/sca-cnn.cvpr17/data
source activate sca

sudo rm -r /home/ml/anaconda3/envs/sca/lib/python2.7/site-packages/numpy

```
还是不行，删除环境重来一遍
```
conda remove -n sca --all
```

train
```
THEANO_FLAGS=mode=FAST_RUN,device=cuda*,floatX=float32 python sca_resnet_branch2b.py
```
then
```
from sklearn.cross_validation import KFold
ImportError: No module named cross_validation
```
install other sklearn veriosn: ref [this](https://scikit-learn.org/dev/versions.html)
```
conda install  scikit-learn=0.19.1
```
not used, ref [this](https://blog.csdn.net/weixin_40283816/article/details/83242083)

then
```
from pycocoevalcap.bleu.bleu import Bleu
ImportError: No module named pycocoevalcap.bleu.bleu
```

### install cocoeval
```
git clone git@github.com:tylin/coco-caption.git

add this in the front of cocoeval.py

import sys
sys.path.append("/home/ml/1_caption_saliency_ref_code/" + 
    "code-and-dataset-for-CapSal/capsal/mrcnn/")

or download the [coco-caption](https://github.com/tylin/coco-caption) for results evaluation. Install it by simply adding it into `$PYTHONPATH`.

```


### Reference
[the difference of py2 and py3](https://www.cnblogs.com/feifeifeisir/p/9599218.html)