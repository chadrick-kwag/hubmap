# kaggle hubmap project

kaggle project: [https://www.kaggle.com/c/hubmap-kidney-segmentation/](https://www.kaggle.com/c/hubmap-kidney-segmentation/)

developed using pytorch 1.5.0, CUDA 10.1


## How to train

the train script is at `train/train_v1.py` and this works with a configuration file where a sample can be found in `train/config/train_v1_config/train_v1.sample.yaml`.

copy the sample config and adjust the options to fit your needs.

then run the train script from bash command like:

```
$ python train_v1.py path/to/config
```

the outputs will be saved in an automatically created directory under `train/ckpt/train_v1`

## Data preparation

data preparation scripts are under `data_prep/`. these scripts are mainly slicing raw data, splitting, restructuring dataset structure, etc.

