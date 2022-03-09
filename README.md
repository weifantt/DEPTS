# DEPTS

Source code for the paper,
["DEPTS: Deep Expansion Learning for Periodic Time Series Forecasting"](https://openreview.net/forum?id=AJAR-JgNw__),
in ICLR22 Spotlight.

## Overview
DEPTS is a customized deep neural network architecture for periodic time series forecasting, which aims to solve the following two challenges:
- To capture diversified periodic compositions
- To model complicated periodic dependencies

## Dataset

You can download the five benchmarks from [Google Drive](https://drive.google.com/file/d/1GYt2chsZLbmJkNG3lb-ytCrI3nuZKDxm/view?usp=sharing). All the datasets are well pre-processed. More details of datasets can be found in the [paper](https://openreview.net/forum?id=AJAR-JgNw__). After downloading the zip file, please unzip it to the root dir of DEPTS for experiments.

## Usage

### Setup
Please use `Python 3(.6)`  as well as the following packages:
```text
torch >= 1.6.0
dataclasses
dtaidistance
pandas
numpy
tqdm
```

### Reproduce
To reproduce the results, you can see more details in `command.sh` and directly run:
```text
sh command.sh
```
Note that all the results reported in the paper are ensembled results of 30 models in order to get a robust evaluation and compare with [N-BEATS](https://arxiv.org/abs/1905.10437). You can also try to run the single model for evaluation if you find it challenging to run all the models.

### Evaluation

To get the evaluation results, run
```text
python evaluation.py
```



## Citation

If you find our work interesting, you can cite the paper as

```text
@inproceedings{
fan2022depts,
title={{DEPTS}: Deep Expansion Learning for Periodic Time Series Forecasting},
author={Wei Fan and Shun Zheng and Xiaohan Yi and Wei Cao and Yanjie Fu and Jiang Bian and Tie-Yan Liu},
booktitle={International Conference on Learning Representations},
year={2022},
url={https://openreview.net/forum?id=AJAR-JgNw__}
}
```
