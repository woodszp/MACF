<div align="center">
  <h1>Multi-granularity Awareness via Gaussian Fusion for Few-Shot Learning</h1>
</div>



## :heavy_check_mark: Requirements
* Ubuntu 20.04
* Python 3.9.6
* [CUDA 11.1](https://developer.nvidia.com/cuda-toolkit)
* [PyTorch 1.7.1](https://pytorch.org)


## :gear: Conda environmnet installation
```bash
conda env create --name MAGF --file environment.yml
conda activate MAGF
```

## :books: Datasets
```bash
cd datasets
bash download_miniimagenet.sh
bash download_cub.sh
bash download_cifar_fs.sh
bash download_tieredimagenet.sh
```

## :pushpin: Quick start: testing scripts
To test in the 5-way K-shot setting:
```bash
bash scripts/test/{dataset_name}_5wKs.sh
```
For example, to test MAGF on the cub dataset in the 5-way 5-shot setting:
```bash
bash scripts/test/cub_5w5s.sh
```
```
python test.py -dataset cub -datadir /home/data/cub -gpu 0 -extra_dir cub_5w5s -temperature_attn 5.0 
```


## :fire: Training scripts
To train in the 5-way K-shot setting:
```bash
bash scripts/train/{dataset_name}_5wKs.sh
```
For example, to train MAGF on the CIFAR-FS dataset in the 5-way 1-shot setting:
```bash
bash scripts/train/cifar_fs_5w1s.sh
```
```
python train.py -batch 64 -dataset cifar_fs -datadir /home/data/cifar_fs -gpu 0 -extra_dir your_run_set -temperature_attn 5.0 -lamb 0.5
```

