# Parkinson's Disease Detection with Hierarchical Attention

This project implements a deep learning pipeline for detecting and classifying Parkinson’s Disease (PD) using wearable sensor data. The model leverages **windowed time-series signals**, **patient metadata/questionnaires**, and a **hierarchical attention architecture**.

---

## 🚀 Features
- **Preprocessing:**
  - Downsampling (100Hz → 64Hz)
  - Bandpass filtering (0.1Hz – 20Hz)
  - Sliding-window segmentation with patient-specific overlap
- **Dataset Handling:**
  - Patient-level splits (train/val/test, K-Fold cross-validation)
  - Each patient includes multiple tasks (e.g., CrossArms, DrinkGlas, PointFinger, etc.)
  - Metadata + questionnaires converted into text embeddings
- **Model:**
  - CNN backbone for time-series feature extraction
  - Multihead self-attention (via PyTorch `nn.MultiheadAttention`)
  - Hierarchical decision-making: 
    - **HC vs PD**
    - **PD vs DD**
- **Training:**
  - Weighted loss functions to handle class imbalance
  - Separate accuracy/precision/recall/F1 metrics for both tasks
  - Patient-level validation (prevents data leakage)

---

## 📂 Project Structure
