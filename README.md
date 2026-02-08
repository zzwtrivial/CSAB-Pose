# Enhancing Low-Light Human Pose Estimation via Multi-Branch Cross-Scale Attention

[![DOI](https://zenodo.org/badge/1151267378.svg)](https://doi.org/10.5281/zenodo.18523011)

> **Note:** This is the official implementation of the manuscript **"Enhancing Low-Light Human Pose Estimation via Multi-Branch Cross-Scale Attention"**

## 📖 Introduction

This repository contains the official PyTorch implementation of our paper. We propose a **Task-Oriented Multi-branch Collaborative Enhancement Framework** for Low-Light Human Pose Estimation (LL-HPE). 

Unlike generic enhancement methods that prioritize visual aesthetics, our method is designed to bridge the domain gap between low-light inputs and frozen pose estimators.

**Core Features:**
- **HVI Decomposition:** Explicitly decouples enhancement into illumination, structure, and noise suppression.
- **Cross-Scale Attention:** Aggregates contextual information to detect small-scale keypoints.
- **Plug-and-Play:** Works with pre-trained estimators (e.g., HRNet, OpenPose) without fine-tuning.
- **Zero-Shot Generalization:** Generalizes to dark scenarios (COCO-Dark) using only normal-light training.

## 🛠️ Dependencies

This project is developed using Python 3.8 and PyTorch 1.10+.

### Installation

1. **Clone this repository:**
   ```bash
   git clone https://github.com/zzwtrivial/CSAB-Pose.git
   cd CSAB-Pose
2. **Create a conda environment:**
```bash
conda create -n llhpe python=3.8
conda activate llhpe

```


3. **Install dependencies:**
```bash
# Install PyTorch
pip install torch torchvision torchaudio --extra-index-url [https://download.pytorch.org/whl/cu113](https://download.pytorch.org/whl/cu113)

4. **Install requirement** 

   This repo require following dependences.

  - **PyTorch >= 0.4.1**
  - numpy >= 1.7.1
  - scipy >= 0.13.2
  - python-opencv >= 3.3.1
  - tqdm > 4.11.1
  - skimage >= 0.13.1

5. Training

```
cd ROOT_DIR/MODEL_DIR/
python3 train.py
```

## 📂 Data Preparation

### ExLPose Dataset

Please download the **ExLPose** benchmark from [Official Link or Repository].
Organize the dataset as follows:

```text
data/
└── ExLPose/
    ├── train/
    │   ├── low/
    │   └── normal/
    └── test/
        ├── low/
        └── normal/

```

### COCO-Dark (Zero-Shot)

The COCO-Dark dataset is synthesized from MS COCO val2017. You can generate it using the script provided in `tools/generate_coco_dark.py`.

## 🧩 Key Algorithms & Code Structure

Our method explicitly decouples the enhancement process. The core implementations can be found in the following files:

* **HVI Decomposition & Reconstruction:**
* File: `lib/models/hvi_module.py` (Example path)
* Description: Implements the conversion between RGB and HVI color spaces and the initial decomposition.


* **Cross-Scale Attention Block (CSAB):**
* File: `lib/models/attention.py`
* Description: The core attention mechanism that aggregates multi-scale features to preserve structural details.


* **Low-Rank Denoising Branch:**
* File: `lib/models/denoise_branch.py`
* Description: Implements the learnable low-rank strategy to suppress sensor noise while preserving edges.


* **Illumination Adjustment Branch:**
* File: `lib/models/illumination.py`
* Description: Predicts the enhancement curve for the V-component.



## 🚀 Usage Guidelines

### 1. Inference Demo

Run visualization on test images:

```bash
cd 256.192.model
python render.py --img_path ../test-lmdb/ --output_path ../visualize/results/
```

* **Note:** Adjust paths according to your data location.

### 2. Training

To train the enhancement module on the ExLPose dataset:

```bash
cd 256.192.model
python train_ll.py
```

* **Config file:** Configurations are in `train_ll_cpn_config.py`.
* **Logs:** Training logs will be saved in `checkpoint/`.

### 3. Testing & Evaluation

Evaluate the model performance on ExLPose dataset:

```bash
cd 256.192.model
python test.py -t 'CPN256x192'
```

* **Pre-trained models:** Place models in `checkpoint/` directory.
* **Results:** Evaluation results will be saved in `result_LL_all/`.

## 📜 Citation



```

```
