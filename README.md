```markdown
# Top-View Person Re-Identification (TVRID)

This repository contains the code for a Top-View Person Re-Identification system. It leverages deep metric learning and classification objectives to match identities across RGB and Depth camera modalities.

The project is built with **PyTorch Lightning** for scalable training and **Hydra** for flexible configuration management.

## ✨ Features
* **Multi-Modal Support:** Train and evaluate on `rgb`, `depth`, or `cross` (RGB-D) tracks.
* **Multiple Backbones:** Easily switch between state-of-the-art feature extractors:
  * **OSNet** (Lightweight, ReID-specific)
  * **ResNet50** (Standard CNN)
  * **ConvNeXt-Tiny** (Modern CNN)
  * **ViT-B/16** (Vision Transformer)
* **Robust Augmentations:** Includes Random Crop, Horizontal Flip, and Random Erasing to improve generalization on top-view images.
* **Two-Stage Evaluation:** Uses a K-Nearest Neighbors (KNN) approach via Euclidean distance to rank gallery images against query images.

## ⚙️ Installation & Requirements

Ensure you have Python 3.8+ installed. Install the required dependencies:

```bash
pip install torch torchvision torchaudio
pip install lightning hydra-core pandas numpy pillow tqdm
```

*(Optional)* To use the OSNet backbone, you must install `torchreid`:
```bash
pip install torchreid
```

## 📂 Dataset Structure
The dataset should be placed in the `data/DB_extracted` directory with the following structure:

```text
data/
└── DB_extracted/
    ├── train/                  # Training images
    ├── test_public/            # Public evaluation images
    ├── train_labels.csv        # Labels for the training set
    └── public_test_labels.csv  # Labels for the test set
```

## 🚀 Training

To train the model, run `train2.py`. You can override configurations directly from the command line using Hydra syntax. 

**Basic Training (ResNet50 on RGB):**
```bash
python train2.py model.backbone_type=resnet track=rgb data.batch_size=4
```

**Memory-Safe Training (Windows):**
If you encounter `[WinError 1455] The paging file is too small`, force data loading to the main process by setting `num_workers=0`:
```bash
python train2.py model.backbone_type=convnext track=depth data.batch_size=4 data.num_workers=0
```

### Supported Backbones
Change the `model.backbone_type=` argument to one of the following:
* `osnet`
* `resnet`
* `vit`
* `convnext`

## 🔍 Evaluation

Evaluation is a two-step process: generating the ranking CSV, and then scoring it.

### 1. Generate Rankings (`eval_generate2.py`)
This script loads a trained checkpoint, extracts features for the test set, computes distances, and outputs a ranked CSV file.

```bash
python eval_generate2.py ^
  --checkpoint lightning_logs/version_X/checkpoints/best.ckpt ^
  --track rgb ^
  --backbone resnet ^
  --data-root data/DB_extracted ^
  --labels-csv data/DB_extracted/public_test_labels.csv ^
  --output outputs/rankings_rgb.csv ^
  --batch-size 4 ^
  --num-workers 0
```
*(Note: Use `--eval-subdir train` and `--labels-csv train_labels.csv` if you are sanity-checking on your training set).*

### 2. Calculate Score (`eval_score.py`)
Compare your generated rankings against the ground truth to get your mAP and Rank-1 metrics.

```bash
python eval_score.py ^
  --rankings outputs/rankings_rgb.csv ^
  --secret-map data/DB_extracted/public_test_labels.csv
```

---

## 📊 Results & Metrics

| Track | Backbone | Image Size | Loss Formulation | mAP (%) | Rank-1 (%) | Rank-5 (%) |
| :--- | :--- | :---: | :--- | :---: | :---: | :---: |
| **RGB** | OSNet | 224x224 | Triplet Loss | 0.88 | 0.82 | 1.0 |
| **RGB** | ResNet50 | 224x224 | ID (CrossEntropy) | 0.91 | 0.94 | 1.0 |
| **RGB** | ConvNeXt-T | 224x224 | ID (CrossEntropy) | 0.92 | 0.96 | 1.0 |
| **RGB** | ViT-B/16 | 224x224 | ID (CrossEntropy) | 0.95 | 0.98 | 1.0 |
| | | | | | | |
| **Depth** | OSNet | 224x224 | Triplet Loss | 0.94 | 0.97 | 1.0 |
| **Depth** | ResNet50 | 224x224 | ID (CrossEntropy) |  0.96 | 0.97 | 1.0 |



## 🛠️ Troubleshooting
* **CUDA Out of Memory:** Reduce `data.batch_size` (e.g., to 4 or 2). ResNet and ConvNeXt are much heavier than OSNet.
* **AssertionError (ViT):** Ensure your image transform resizes to `224x224`. ViT strictly requires this dimension, unlike CNNs.
* **0% Accuracy during generation:** Ensure the `--track` flag in `eval_generate2.py` matches the modality you trained on.
```
