<div align="center">
    <h2>
        Prompt-Driven Lightweight Foundation Model for Instance Segmentation-Based Fault Detection in Freight Trains
    </h2>
</div>
<br>

## Introduction

This repository is the code implementation of the paper Prompt-Driven Lightweight Foundation Model for Instance Segmentation-Based Fault Detection in Freight Trains, which is based on the [MMDetection](https://github.com/open-mmlab/mmdetection/tree/main) project.

## Installation

### Dependencies

- Linux or Windows
- Python 3.7+, recommended 3.10
- PyTorch 2.0 or higher, recommended 2.1
- CUDA 11.7 or higher, recommended 12.1
- MMCV 2.0 or higher, recommended 2.1

### Environment Installation
<details open>

**Step 0**: Install [Miniconda](https://docs.conda.io/projects/miniconda/en/latest/index.html).

**Step 1**: Create a virtual environment named `ProLIFT` and activate it.

```shell
conda create -n ProLIFT python=3.10 -y
conda activate ProLIFT
```

**Step 2**: Install [PyTorch2.1.x](https://pytorch.org/get-started/locally/).

Linux/Windows:
```shell
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121
```
Or

```shell
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=12.1 -c pytorch -c nvidia
```

**Step 3**: Install [MMCV2.1.x](https://mmcv.readthedocs.io/en/latest/get_started/installation.html).

```shell
pip install -U openmim
mim install mmcv==2.1.0
```

**Step 4**: Install other dependencies.

```shell
pip install -U transformers==4.38.1 wandb==0.16.3 einops pycocotools shapely scipy terminaltables importlib peft==0.8.2 mat4py==0.6.0 mpi4py
```

**Step 5**: [Optional] Install DeepSpeed.

If you want to use DeepSpeed to train the model, you need to install DeepSpeed. The installation method of DeepSpeed can refer to the [DeepSpeed official document](https://github.com/microsoft/DeepSpeed).

```shell
pip install deepspeed==0.13.4
```

Note: The support for DeepSpeed under the Windows system is not perfect yet, we recommend that you use DeepSpeed under the Linux system.

## Model Training
#### Single Card Training

```shell
sh train.sh
```

#### Multi-card Training

```shell
sh multi_train.sh
```