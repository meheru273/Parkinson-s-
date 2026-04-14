# Config Summary — `results-new/`

All runs share: `data_root=PADS 1.0.0`, `apply_downsampling=true`, `input_dim=6` (except CNN/LSTM), `timestep`/`window_size=256`, `num_folds=5`.

## 1. BaseModel + BandPass
File: [BaseModel_wBandPass_wDownsampling/best_model_config.json](BaseModel_wBandPass_wDownsampling/best_model_config.json)

Transformer (from initial CV HPs): `model_dim=64, num_heads=8, num_layers=3, d_ff=256, dropout=0.2, bs=32, lr=5e-4, wd=0.01, epochs=100`. `bandpass=true`, `num_classes=2`. Best fold-0 epoch 37 → 0.9303 combined.

## 2. BaseModel without BandPass
File: [BaseModel_woutBandPass_wDownsampling/best_model_fold_1_config.json](BaseModel_woutBandPass_wDownsampling/best_model_fold_1_config.json)

Same Transformer HPs as #1 but `bandpass=false`. Fold-0 best epoch 26 → 0.8657 combined. Folds 1–5 configs identical apart from `best_epoch`/`best_val_acc`.

## 3. ThreeClass BaseModel
File: [ThreeClass_BaseModel/output_3class/best_model_fold_1_config.json](ThreeClass_BaseModel/output_3class/best_model_fold_1_config.json)

Same Transformer HPs as #1; `num_classes=3`, `output_dir=output_3class`. Fold-0 best epoch 52 → 0.8964.

## 4. SSL BaseModel — Full Fine-tune
File: [SSL_BaseModel/full-finrtune/checkpoints/config.json](SSL_BaseModel/full-finrtune/checkpoints/config.json)

SSL encoder: `model_dim=32, num_heads=8, num_layers=3, d_ff=256, dropout=0.1228, projection_dim=64`.
Pretrain: `mode=ssl, loss=infonce, temperature=0.07, margin=0.3, bs=64, lr=1e-4, epochs=50, aug_strength=0.3, aug_type=3`.
Finetune: `evaluation_type=finetune, bs=32, lr=2.91e-4, wd=1.62e-4, epochs=50, patience=10`.
Label fractions: `[0.05, 0.1, 0.2, 0.5, 0.7, 1.0]`. `label_efficiency_config.json` is identical.

## 5. SSL BaseModel — Linear Probe
File: [SSL_BaseModel/linear-prob/checkpoints/config.json](SSL_BaseModel/linear-prob/checkpoints/config.json)

Identical to #4 except `evaluation_type=linear`.

## 6. TimesFM LoRA — Single Fold
File: [TimesFM_LoRA/results/timesfm_lora/checkpoints/config.json](TimesFM_LoRA/results/timesfm_lora/checkpoints/config.json)

`model_dim=1280, hidden_dim=512, dropout=0.3, label_smoothing=0.1, lora_r=6, lora_alpha=12, lora_dropout=0.15, mixup_alpha=0.1, bs=32, lr=1e-4, wd=0.01, epochs=40, warmup=3, early_stop=10, max_folds_to_train=1, ablation_type=lora`.

## 7. TimesFM LoRA — K-Fold
File: [TimesFM_KFOLD/results/timesfm_lora/checkpoints/config.json](TimesFM_KFOLD/results/timesfm_lora/checkpoints/config.json)

Identical to #6 except **`max_folds_to_train=5`**.

## 8. CNN-LSTM Task-Wise Ablation
File: [lstm_cnn_ablation_taskwise/checkpoints/config.json](lstm_cnn_ablation_taskwise/checkpoints/config.json)

`window_size=256, input_channels=12, hidden_size=128, num_lstm_layers=2, bidirectional=true, dropout=0.3, bs=32, lr=1e-3, wd=1e-4, epochs=50`.
Tasks: CrossArms, DrinkGlas, Entrainment, HoldWeight, LiftHold, PointFinger, Relaxed, StretchHold, TouchIndex, TouchNose. `model_types=[cnn, lstm]`.

## 9. Nested CV Transformer
File: [nested-cv/summary/config.json](nested-cv/summary/config.json)

`outer_folds=5, inner_folds=3, n_trials=20, optuna_epochs=20, final_epochs=80, num_classes=2, patience=10, checkpoint_interval=5`. Resumed from a prior Kaggle notebook snapshot. Per-fold best HPs are listed in [results.md](results.md) §8.

## 10. `ssl-results/` Per-Fold (fold_0 … fold_3)
File: [ssl-results/fold_0/checkpoints/config.json](ssl-results/fold_0/checkpoints/config.json)

Same SSL Transformer shape as #4, but **`augmentation_type=2`** and **`loss_type=ntxent`** (vs `aug_type=3, infonce` in SSL_BaseModel). `evaluation_type=finetune`. Folds 1–3 identical.

## 11. `ssl-results/linear_eval_fold_0`
File: [ssl-results/linear_eval_fold_0/checkpoints/config.json](ssl-results/linear_eval_fold_0/checkpoints/config.json)

Matches SSL_BaseModel linear-probe (`aug_type=3, infonce, evaluation_type=linear`) — **not** the ntxent variant used by fold_0…fold_3 finetune.

## 12. `ssl-results/linear_probe` — Per-Fraction
File: [ssl-results/linear_probe/checkpoints/label_eff_100pct_config.json](ssl-results/linear_probe/checkpoints/label_eff_100pct_config.json)

Slightly different schema: model HPs appended at bottom — `model_dim=64, num_heads=8, num_layers=3, d_ff=256, dropout=0.2, bs=32, lr=5e-4, wd=0.01, hp_source=initial_cv`. `loss=infonce, aug_type=3, evaluation_type=finetune` (says `finetune` despite being under `linear_probe/`). Fractions 5/10/20/50/70/100 have analogous configs.

---

## Notable Discrepancies

- **SSL augmentation/loss mismatch**: `SSL_BaseModel/*` and `ssl-results/linear_eval_fold_0` use `aug_type=3 + infonce`, while `ssl-results/fold_0…fold_3` (finetune folds) use `aug_type=2 + ntxent`. Two different SSL pretraining recipes.
- **SSL linear-probe encoder sizes differ**: `SSL_BaseModel/linear-prob` uses `model_dim=32, dropout=0.1228, lr=2.91e-4`, while `ssl-results/linear_probe/label_eff_*` uses `model_dim=64, dropout=0.2, lr=5e-4` (initial-CV Transformer HPs). The stronger linear-probe numbers in [results.md](results.md) §5 come from the larger encoder.
- **TimesFM LoRA**: only difference between `TimesFM_LoRA` and `TimesFM_KFOLD` is `max_folds_to_train` (1 → 5) — same LoRA hyperparameters.
- **Nested-CV config** does not store HPs directly; per-fold best HPs were recovered from Optuna output and live in [results.md](results.md) §8.
