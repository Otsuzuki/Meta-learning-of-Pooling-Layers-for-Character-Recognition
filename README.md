# Meta-learning of Pooling Layers for Character Recognition

Code and meta-parameters corresponding to Meta-learning of Pooling Layers for Character Recognition, found at this arXiv link.
 
# Features
 
"hoge"のセールスポイントや差別化などを説明する

<div align="center">
<img src=./figures/PoolingComparison.png alt="属性" title="タイトル">
</div>

<!-- <img src=./figures/figure2-7.jpg width="460px"> -->
 
# Installation and requirements
 
All experiments were run in a conda environment with python 3.7.4, using pytorch 1.4.0. The conda environment we used is exported in environment.yml.
 
# Simulation of Artificial 1D data
## Generate data
The file ```train.py``` contains the data generation code. You must specify the name of the problem we wish to generate data for. Options are:

* ```Max_1d``` : 1-D data generated by a max pooling layer
* ```Average_1d``` : 1-D data generated by a average pooling layer
* ```Square_1d``` : 1-D data generated by the pooling layer consists of the max pooling for the first half and average pooling for the second half
* ```Max_2d``` : 2-D data generated by a max pooling layer
* ```Average_2d``` : 2-D data generated by a average pooling layer
* ```Square_2d``` : 2-D data generated by the pooling layer consists of the max pooling for the first half and average pooling for the second half
* ```NonSquare_2d```  
  Example:  
```python train.py --problem Half_maxavg```
 
# Character Recognition (Omniglot)
 
WIP
 
# Author 
* Author : Takato Otsuzuki, Hideaki Hayashi, Heon Song, Seiichi Uchida
* Affiliation : Kyushu University, Fukuoka, Japan
* Contact E-mail : takato.otsuzuki@human.ait.kyushu-u.ac.jp, hayashi@ait.kyushu-u.ac.jp
