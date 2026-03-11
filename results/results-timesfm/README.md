# Parkinson's Disease Detection with PADS Dataset-
# Parkinson's- SmartWatch Dataset
# Cross Attention transformer encoder architecture
# Nested Cross validation
# inference in edge device using raspberry pi
# self supervised Learning 
# Parkinson's Disease Detection Scripts


This repository contains two main scripts for Parkinson's disease detection using wrist sensor data. Both scripts support advanced model architectures, data augmentation, and robust evaluation protocols.

## scripts/self_supervise_base_model.py

**Self-Supervised Contrastive Learning Pipeline**

- Implements self-supervised pre-training using contrastive loss (contrastive, triplet, NT-Xent, InfoNCE).
- Fine-tuning and evaluation with full metrics, ROC curves, and t-SNE plots.
- Data augmentation: Gaussian noise, time warping, scaling, permutation, and more.
- Stratified K-Fold cross-validation at the patient level.
- Patient-level splitting and metrics saving per fold.

**Key Model Configurations:**
```
input_dim: 6
model_dim: 32
num_heads: 8
num_layers: 3
d_ff: 256
dropout: 0.12
timestep: 256
projection_dim: 64
pretrain_batch_size: 64
pretrain_lr: 1e-4
pretrain_epochs: 50
loss_type: contrastive | triplet | ntxent | infonce
negative_sampling_strategy: random | hard | hierarchical
batch_size: 32
learning_rate: ~3e-4
weight_decay: ~1.6e-4
num_epochs: 40
early_stopping_patience: 15
```
See the `config` dictionary in the script for all options.

## scripts/timesfm_ablation.py

**TimesFM Ablation Study Pipeline**

- Uses the TimesFM model as a feature extractor for classification.
- Supports two main fine-tuning strategies:
	- Gradual unfreezing (with discriminative learning rates for backbone and heads)
	- LoRA (Low-Rank Adaptation) adapters for efficient fine-tuning
- Checkpointing, metrics saving after each epoch, and visualization utilities (loss, ROC, t-SNE).

**Key Model Configurations:**
```
model_dim: 1280  # TimesFM hidden size
hidden_dim: 512
dropout: 0.1
lora_r: 8
lora_alpha: 16
lora_dropout: 0.1
num_folds: 5
max_folds_to_train: 1
batch_size: 32
learning_rate: 1e-4 (or 5e-4 for LoRA)
weight_decay: 1e-4
num_epochs: 50
backbone_lr: 2e-5 (for gradual unfreezing)
head_lr: 1e-3 (for gradual unfreezing)
warmup_epochs: 5-10
unfreeze_after_epoch: 10
unfreeze_n_layers: 2
```
See the `base_config` and per-ablation-type overrides in the script for all options.

---


---

**Requirements:**
- Python 3.x
- PyTorch, scikit-learn, numpy, matplotlib, tqdm, scipy


**Usage:**
1. Edit configuration parameters inside each script as needed (see the `config` dictionary in each script).
2. Run the scripts directly to start pre-training, fine-tuning, or ablation studies:
	- `python scripts/self_supervise_base_model.py`
	- `python scripts/timesfm_ablation.py`

Results, metrics, and plots will be saved to the output directories specified in the configs.

See comments and configuration blocks in each script for further details and customization.
