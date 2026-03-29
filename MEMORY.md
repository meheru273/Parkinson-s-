# Parkinson's Disease Detection Project - Memory

## Project Overview
A PyTorch-based machine learning project detecting Parkinson's Disease from smartwatch IMU sensor data using Dual-Channel Transformer architecture.

## Key Architecture
- **Dual-head hierarchical classification**: HC vs PD (2-class) + PD vs DD (2-class)
- **Best model**: Cross-attention transformer between left/right wrist channels
- **Best accuracy**: 96.84% combined (HC vs PD: 97.10%, PD vs DD: 96.58%)
- **Edge deployment**: ~47ms latency on Raspberry Pi

## Main Scripts & Components
- `scripts/base_model.py` — Base dual-channel transformer (96.84% accuracy)
- `scripts/nested_cv_base_model.py` — Optuna hyperparameter optimization (88.78% accuracy, edge-optimized)
- `scripts/self_supervise_base_model.py` — Contrastive learning pre-training (95.16% accuracy)
- `scripts/timesfm_ablation.py` — Foundation model LoRA fine-tuning (91.51% accuracy)
- `scripts/three_class_base_model.py` — Multi-class baseline (76.17% accuracy - underperforming)
- `components/dataloader.py` — Data preprocessing (downsample 100Hz→64Hz, bandpass filter 0.1-20Hz, 256-sample windows)

## Latest Experiments
- Modified: `scripts/nested_cv_base_model.py`, `scripts/three_class_base_model.py`
- Results: `/results-new/SSL_BaseModel/`, `/results-new/TimesFM_LoRA/`, etc.

## Dataset
- PADS v1.0.0: Bilateral wrist smartwatch IMU (6-axis per wrist)
- 10 motor tasks (CrossArms, DrinkGlas, etc.)
- 3 cohorts: HC (Healthy Controls), PD (Parkinson's), DD (Differential Diagnosis)
- Output: important.md created with comprehensive structure documentation
