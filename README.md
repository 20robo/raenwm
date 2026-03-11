# RAE-NWM: Navigation World Model in Dense Visual Representation Space

[**Paper**](https://arxiv.org/abs/2603.09241) | [**Models**](#) *(Coming Soon)*

> **Note:** This repository contains the official implementation of RAE-NWM. 
> The pre-trained RAE-NWM weights and complete model cards will be released here soon.

## 📥 Installation & Setup

First, clone the repository and navigate into the project directory:

```bash
git clone https://github.com/zmkun20/raenwm.git
cd raenwm
```

## ⚙️ Environment Setup

We recommend using Anaconda to manage the environment:

```bash
conda create -y -n raenwm python=3.11.10
conda activate raenwm

# Install PyTorch
pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu128

# Install dependencies
conda install -y ffmpeg
pip install "numpy<2" omegaconf huggingface_hub decord einops evo transformers diffusers tqdm timm notebook dreamsim torcheval lpips ipywidgets accelerate==0.23.0 torchdiffeq==0.2.5 wandb
```

## 📂 Data Preparation

To download and preprocess data, please follow the steps from [NoMaD](https://github.com/robodhruv/visualnav-transformer?tab=readme-ov-file#data-wrangling). 
Specifically, after downloading the datasets, run `process_bags.py` and `process_recon.py` to save each processed dataset to `path/to/nwm_repo/data/<dataset_name>`.

Your final data directory structure should look like this:

```text
raenwm/data
└── <dataset_name>
    ├── <name_of_traj1>
    │   ├── 0.jpg
    │   ├── ...
    │   ├── T_1.jpg
    │   └── traj_data.pkl
    ├── <name_of_traj2>
    │   ├── 0.jpg
    │   ├── ...
    │   └── traj_data.pkl
    └── <name_of_trajN>
        ├── 0.jpg
        ├── ...
        └── traj_data.pkl
```

## 📦 Model Weights

### 1. RAE Weights (DINOv2)
We follow the official [RAE](https://github.com/bytetriper/RAE) instructions to download the DINOv2 decoder weights and normalization stats:

```bash
huggingface-cli login
mkdir -p models
huggingface-cli download nyu-visionx/RAE-collections \
  --repo-type model \
  --include "decoders/dinov2/wReg_base/ViTXL_n08/model.pt" \
  --include "stats/dinov2/wReg_base/imagenet1k/stat.pt" \
  --local-dir ./models
```

### 2. RAE-NWM Weights
⏳ **[Coming Soon]** The pre-trained weights for our model will be released and linked here.

## 💻 Multi-GPU Support

This repository natively supports multi-GPU operations for both training and inference. To utilize multiple GPUs, simply replace `python` with `torchrun` and specify the number of processes. For example:

```bash
torchrun --nproc_per_node=8 train.py ...
```

## 🚀 Training

To train the model from scratch, run:

```bash
python train.py \
  --config config/raenwm.yaml \
  --epochs 50 \
  --global-seed 42 \
  --log-every 100 \
  --ckpt-every 5000 \
  --eval-every 1000 \
  --bfloat16 1 \
  --torch-compile 1
```

## 📊 Inference & Evaluation

We provide a streamlined bash script to run the evaluation pipeline, which supports both `time` and `rollout` modes. The script allows you to selectively run ground-truth preparation (`gt`), inference (`infer`), evaluation (`eval`), or all steps at once (`all`).

```bash
# Usage: bash run_eval.sh [MODE] [STEP] [DATASET] [CKPT_PATH]

# Example: Generate Ground-Truth only
bash run_eval.sh time gt sacson path/to/checkpoints/checkpoint.pth.tar

# Example: Run Inference only
bash run_eval.sh time infer sacson path/to/checkpoints/checkpoint.pth.tar

# Example: Run all steps sequentially for rollout mode
bash run_eval.sh rollout all sacson path/to/checkpoints/checkpoint.pth.tar
```

## 🛣️ Planning

To evaluate planning performance using CEM, simply run the provided planning script:

```bash
# Usage: bash run_plan.sh [DATASET] [curve | line] [CKPT_PATH]
bash run_plan.sh sacson curve path/to/checkpoints/checkpoint.pth.tar
```

## 🔬 Probe Experiment

To reproduce the representation analysis (probe experiment) mentioned in the paper, we provide the following scripts. 

**Train and Evaluate:**
Training the probe will automatically run the evaluation at the end of the process:
```bash
python train_probe.py \
  --config config/probe.yaml \
  --epochs 5 \
  --global-seed 42 \
  --log-every 100 \
  --ckpt-every 1000 \
  --eval-every 1000 \
  --bfloat16 1 \
  --torch-compile 0
```

## 🙏 Acknowledgements

Our project is inspired by and built upon [RAE](https://github.com/bytetriper/RAE), [NWM](https://github.com/facebookresearch/nwm), and [NoMaD](https://github.com/robodhruv/visualnav-transformer).