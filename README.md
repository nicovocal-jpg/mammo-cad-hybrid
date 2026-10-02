# Hybrid CAD System for Mammographic Lesions

**InceptionV3 classification + YOLOv8 detection on DDSM, with a leakage-free patient-level split.**

Code for the paper *"Hybrid System for Computer-Aided Classification and Detection of Mammographic Lesions"*, accepted at **IEEE SIME 2026** (Sousse, Tunisia, Nov 2–4, 2026, Paper ID #258).

![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-FF6F00?logo=tensorflow&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-Ultralytics_YOLOv8-EE4C2C?logo=pytorch&logoColor=white)
![Status](https://img.shields.io/badge/IEEE_SIME_2026-accepted-2ea44f)

---

## Overview

A two-stage computer-aided system for screening mammography:

| Stage | Model | Task | Output |
|---|---|---|---|
| 1 | **InceptionV3** (ImageNet transfer learning + two-stage fine-tuning) | Image-level classification | `normal` / `mass` / `calcification` |
| 2 | **YOLOv8** (Ultralytics) | Lesion localization | Bounding boxes (`mass`, `calcification`) |

```
mammogram ──► InceptionV3 ──► class + probability
          └─► YOLOv8      ──► lesion boxes
```

## Results

| Model | Split | Metric | Value |
|---|---|---|---|
| InceptionV3 | Image-level (paper, original) | Accuracy / AUC | 83.15 % / 95.49 % |
| YOLOv8m | Image-level (paper, original) | Precision / Recall / mAP@50 | 79.55 % / 77.21 % / 74.80 % |
| InceptionV3 | **Patient-level (re-evaluation)** | Accuracy | **73.99 %** (test n = 961) |
| YOLOv8 | Patient-level, 2 classes + background | — | re-evaluation in progress |

### Why there are two sets of numbers

Peer review flagged the most important weakness of the first version: the DDSM split was done **per image, not per patient**, so views of the same patient could end up in both train and test (data leakage). I rebuilt the dataset to fix it:

- **CBIS-DDSM** for `mass` / `calcification` (from `metadata_with_jpg_img_.csv`) + **Mini-DDSM** for `normal` (`Status == 'Normal'`).
- A **patient-level split manifest** (`patient_level_split_manifest.csv`). This resolved the 31 CBIS-DDSM patients who appeared in both the official train and test sets.
- The detection set now includes **background (normal) images** so false positives are measured too.

As expected, accuracy dropped once the leakage was removed. The patient-level figure is the honest estimate of how the model generalizes.

## Reproducibility

Both scripts are written to answer the reviewers' requests:

- **Single global seed** (`SEED = 42`) applied to Python, NumPy and TensorFlow, with deterministic ops when available.
- Every hyperparameter the paper did not state (dropout, learning rates, patience, unfreeze layer) is declared as a named constant and saved to `run_metadata.json`.
- The **real early-stopping epoch** is logged for each stage.
- **All test-set predictions** are saved (`test_predictions.csv`, YOLO `save_json`), so bootstrap confidence intervals, calibration, FROC and Grad-CAM can be computed without retraining.
- **Hardware-aware config:** on GPUs with less than 6 GB of VRAM, the YOLO script falls back to YOLOv8s at 512 px and logs the deviation.

## Repository structure

```
├── train_inceptionv3.py   # Classification: 2-stage transfer learning + full test predictions
├── train_yolov8.py        # Detection: 2-stage training + test evaluation (P-R, F1, confusion matrix)
├── inception.ipynb        # Executed classification run (patient-level split)
├── LICENSE
├── requirements.txt
└── CITATION.cff
```

## Usage

```bash
conda create -n mammo-cad python=3.10 -y
conda activate mammo-cad
pip install -r requirements.txt

# 1. Edit DATASET_ROOT / DATA_YAML / OUTPUT_DIR at the top of each script
# 2. Train
python train_inceptionv3.py
python train_yolov8.py
```

Expected dataset layout:

```
dataset_final/{train,val,test}/{calcification,mass,normal}/*.jpg
detection_dataset/data.yaml   # YOLO format, classes: 0=mass, 1=calcification
```

## Data

No images are included in this repository. Download them from their sources:

- **CBIS-DDSM**: [The Cancer Imaging Archive](https://www.cancerimagingarchive.net/collection/cbis-ddsm/)
- **Mini-DDSM**: public subset of DDSM (search "Mini-DDSM" on Kaggle)

## Authors

**Nicolás Villarroel Vocal**, Rodrigo Martínez Severich, Edgar R. Ramos Silvestre, Eynar Calle Vives
Universidad Privada del Valle (UNIVALLE), Cochabamba, Bolivia

## Citation

```bibtex
@inproceedings{villarroel2026hybrid,
  title     = {Hybrid System for Computer-Aided Classification and Detection of Mammographic Lesions},
  author    = {Villarroel Vocal, Nicol{\'a}s and Mart{\'i}nez Severich, Rodrigo and Ramos Silvestre, Edgar R. and Calle Vives, Eynar},
  booktitle = {Proceedings of IEEE SIME 2026},
  year      = {2026}
}
```

> ⚠️ Research code. This is not a medical device and is not intended for clinical use.

## License

Code released under the MIT License (see `LICENSE`). The paper is © IEEE.
