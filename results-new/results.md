# Parkinson's Disease Classification — Results Summary

---

## 1. BaseModel with BandPass + Downsampling

### HC vs PD (Binary)

| Fold | Accuracy | Precision | Recall | F1 |
|------|----------|-----------|--------|----|
| 1    | 0.9355   | 0.9421    | 0.9355 | 0.9347 |
| 2    | 0.9347   | 0.9408    | 0.9347 | 0.9339 |
| 3    | 0.9307   | 0.9352    | 0.9307 | 0.9290 |
| 4    | 0.9301   | 0.9345    | 0.9301 | 0.9290 |
| 5    | 0.9251   | 0.9345    | 0.9251 | 0.9245 |
| **Mean ± Std** | **0.9312 ± 0.0043** | **0.9374 ± 0.0037** | **0.9312 ± 0.0043** | **0.9302 ± 0.0046** |

### PD vs DD (Binary)

| Fold | Accuracy | Precision | Recall | F1 |
|------|----------|-----------|--------|----|
| 1    | 0.9251   | 0.9346    | 0.9251 | 0.9241 |
| 2    | 0.8810   | 0.8823    | 0.8810 | 0.8807 |
| 3    | 0.8233   | 0.8233    | 0.8233 | 0.8233 |
| 4    | 0.8685   | 0.8703    | 0.8685 | 0.8694 |
| 5    | 0.8541   | 0.8575    | 0.8541 | 0.8529 |
| **Mean ± Std** | **0.8704 ± 0.0366** | **0.8736 ± 0.0348** | **0.8704 ± 0.0366** | **0.8701 ± 0.0365** |

---

## 2. BaseModel without BandPass, with Downsampling

### HC vs PD (Binary)

| Fold | Accuracy | Precision | Recall | F1 |
|------|----------|-----------|--------|----|
| 1    | 0.8617   | 0.8627    | 0.8617 | 0.8608 |
| 2    | 0.8802   | 0.8805    | 0.8802 | 0.8798 |
| 3    | 0.8161   | 0.8163    | 0.8161 | 0.8162 |
| 4    | 0.8161   | 0.8163    | 0.8161 | 0.8162 |
| 5    | 0.8648   | 0.8654    | 0.8648 | 0.8649 |
| **Mean ± Std** | **0.8598 ± 0.0274** | **0.8602 ± 0.0273** | **0.8598 ± 0.0274** | **0.8596 ± 0.0275** |

### PD vs DD (Binary)

| Fold | Accuracy | Precision | Recall | F1 |
|------|----------|-----------|--------|----|
| 1    | 0.8623   | 0.8649    | 0.8623 | 0.8617 |
| 2    | 0.8809   | 0.8823    | 0.8809 | 0.8807 |
| 3    | 0.8233   | 0.8233    | 0.8233 | 0.8233 |
| 4    | 0.8541   | 0.8574    | 0.8541 | 0.8530 |
| 5    | 0.8488   | 0.8539    | 0.8488 | 0.8478 |
| **Mean ± Std** | **0.8539 ± 0.0233** | **0.8564 ± 0.0230** | **0.8539 ± 0.0233** | **0.8533 ± 0.0238** |

---

## 3. Three-Class BaseModel (HC vs PD vs DD)

### Per-Fold Results

| Fold | Best Epoch | Overall Acc | HC Acc | PD Acc | DD Acc |
|------|------------|-------------|--------|--------|--------|
| 1    | 50         | 0.8893      | 0.8446 | 1.0000 | 0.8125 |
| 2    | 50         | 0.8796      | 0.7861 | 1.0000 | 0.8398 |
| 3    | 32         | 0.8510      | 0.6923 | 1.0000 | 0.8421 |
| 4    | 50         | 0.8603      | 0.7236 | 0.9958 | 0.8429 |
| 5    | 50         | 0.8883      | 0.8462 | 0.9986 | 0.8071 |
| **Mean ± Std** | — | **0.8737 ± 0.0165** | **0.7786 ± 0.0629** | **0.9989 ± 0.0018** | **0.8289 ± 0.0168** |

### Per-Class Accuracy Summary
- **HC (Healthy Control):** 0.7786 ± 0.0629 — highest variance, most challenging class
- **PD (Parkinson's Disease):** 0.9989 ± 0.0018 — near-perfect, most discriminable
- **DD (Differential Diagnosis):** 0.8289 ± 0.0168 — intermediate difficulty

---

## 4. SSL BaseModel — Full Fine-tune (Label Efficiency)

> Averaged over 3 complete folds (fold_0, fold_1, fold_2). Folds 3–4 are incomplete.

| % Labels | N Samples | HC vs PD Acc | PD vs DD Acc | Combined Acc | HC F1   | PD F1   |
|----------|-----------|--------------|--------------|--------------|---------|---------|
| 5%       | 747       | 0.7939       | 0.7878       | 0.7909       | 0.7925  | 0.7866  |
| 10%      | 1,494     | 0.8703       | 0.8591       | 0.8648       | 0.8684  | 0.8575  |
| 20%      | 2,987     | 0.9226       | 0.9167       | 0.9196       | 0.9216  | 0.9161  |
| 50%      | 7,466     | 0.9293       | 0.9218       | 0.9255       | 0.9284  | 0.9211  |
| 70%      | 10,453    | 0.9344       | 0.9213       | 0.9279       | 0.9337  | 0.9207  |
| 100%     | 14,932    | 0.9356       | 0.9250       | 0.9303       | 0.9348  | 0.9244  |

### Per-Fold at 100% Labels

| Fold | HC vs PD Acc | PD vs DD Acc | Combined Acc |
|------|-------------|-------------|-------------|
| 0    | 0.9351      | 0.9248      | 0.9299      |
| 1    | 0.9359      | 0.9251      | 0.9305      |
| 2    | 0.9359      | 0.9251      | 0.9305      |

> Reaches ~92% combined accuracy with only 20% of labels. Consistent across folds.

---

## 5. SSL BaseModel — Linear Probe (Label Efficiency)

> Updated run with linear evaluation head (frozen SSL encoder). Source: [label_efficiency_results.csv](ssl-results/linear_probe/metrics/label_efficiency_results.csv).

| % Labels | N Samples | HC vs PD Acc | PD vs DD Acc | Combined Acc | HC F1   | PD F1   | Val CSE | Best Epoch |
|----------|-----------|--------------|--------------|--------------|---------|---------|---------|------------|
| 5%       | 1,282     | 0.6564       | 0.6629       | 0.6597       | 0.6564  | 0.6627  | 0.7025  | 42         |
| 10%      | 2,564     | 0.7865       | 0.7475       | 0.7670       | 0.7862  | 0.7475  | 0.4842  | 29         |
| 20%      | 5,128     | 0.8280       | 0.8148       | 0.8214       | 0.8280  | 0.8139  | 0.4015  | 43         |
| 50%      | 12,820    | 0.8905       | 0.8834       | 0.8870       | 0.8902  | 0.8826  | 0.2925  | 45         |
| 70%      | 17,948    | 0.9100       | 0.9048       | 0.9074       | 0.9097  | 0.9046  | 0.2443  | 50         |
| 100%     | 25,640    | **0.9412**   | **0.9390**   | **0.9401**   | 0.9409  | 0.9389  | 0.1502  | 44         |

> Updated linear probe reaches **94.01% combined** at 100% labels — now matching (and slightly exceeding) full fine-tune. Large jump over prior run (~81%); frozen SSL features are competitive given more samples and longer training. At 50% labels, already crosses 88%.

---

## 6. TimesFM LoRA — 4-Fold K-Fold CV

> 4 outer folds available. Best-epoch (by validation accuracy) reported per fold. Source: [TimesFM_KFOLD/results/timesfm_lora/metrics](TimesFM_KFOLD/results/timesfm_lora/metrics/).

### HC vs PD

| Fold | Best Epoch | Accuracy | Precision | Recall | F1     |
|------|------------|----------|-----------|--------|--------|
| 1    | 7          | 0.9371   | 0.9412    | 0.9371 | 0.9365 |
| 2    | 9          | 0.9363   | 0.9424    | 0.9363 | 0.9355 |
| 3    | 8          | 0.9355   | 0.9404    | 0.9355 | 0.9348 |
| 4    | 7          | 0.9361   | 0.9426    | 0.9361 | 0.9353 |
| **Mean ± Std** | — | **0.9362 ± 0.0006** | **0.9416 ± 0.0010** | **0.9362 ± 0.0006** | **0.9355 ± 0.0007** |

### PD vs DD

| Fold | Best Epoch | Accuracy | Precision | Recall | F1     |
|------|------------|----------|-----------|--------|--------|
| 1    | 9          | 0.9278   | 0.9362    | 0.9278 | 0.9272 |
| 2    | 14         | 0.9349   | 0.9391    | 0.9349 | 0.9346 |
| 3    | 8          | 0.9240   | 0.9323    | 0.9240 | 0.9234 |
| 4    | 6          | 0.9290   | 0.9353    | 0.9290 | 0.9283 |
| **Mean ± Std** | — | **0.9289 ± 0.0039** | **0.9357 ± 0.0024** | **0.9289 ± 0.0039** | **0.9284 ± 0.0040** |

> Highly stable on HC vs PD (std ~0.0006). PD vs DD shows more fold variance but still competitive with the nested-CV transformer baseline. Combined mean ≈ 0.9326.

---

## 7. CNN-LSTM Task-Wise Ablation

> 5-fold CV per task, separately for CNN and LSTM architectures.
> Config: window_size=256, hidden_size=128, num_lstm_layers=2, bidirectional=True, dropout=0.3, lr=0.001.

### Aggregated Results (Mean over 5 Folds)

| Task        | CNN Combined | CNN HC Acc | CNN PD Acc | LSTM Combined | LSTM HC Acc | LSTM PD Acc |
|-------------|-------------|-----------|-----------|--------------|------------|------------|
| CrossArms   | 0.8709      | 0.8988    | 0.8431    | 0.9151       | 0.9222     | 0.9080     |
| DrinkGlas   | 0.8617      | 0.9008    | 0.8226    | 0.9154       | 0.9181     | 0.9127     |
| Entrainment | 0.7836      | 0.8076    | 0.7597    | **0.9601**   | 0.9643     | 0.9558     |
| HoldWeight  | 0.8991      | 0.9182    | 0.8800    | 0.9145       | 0.9181     | 0.9108     |
| LiftHold    | 0.8992      | 0.9191    | 0.8793    | 0.9147       | 0.9222     | 0.9072     |
| PointFinger | 0.8058      | 0.8214    | 0.7901    | 0.8998       | 0.9110     | 0.8886     |
| Relaxed     | **0.9357**  | 0.9582    | 0.9132    | 0.9547       | 0.9617     | 0.9477     |
| StretchHold | 0.8883      | 0.9161    | 0.8606    | 0.9154       | 0.9182     | 0.9127     |
| TouchIndex  | 0.8394      | 0.8701    | 0.8088    | 0.8807       | 0.8925     | 0.8689     |
| TouchNose   | 0.7177      | 0.7111    | 0.7243    | 0.8278       | 0.8516     | 0.8041     |
| **Mean**    | **0.8501**  | 0.8721    | 0.8282    | **0.9098**   | 0.9280     | 0.8917     |

### Per-Fold Detail (Combined Accuracy)

| Task        | CNN F1 | CNN F2 | CNN F3 | CNN F4 | CNN F5 | LSTM F1 | LSTM F2 | LSTM F3 | LSTM F4 | LSTM F5 |
|-------------|--------|--------|--------|--------|--------|---------|---------|---------|---------|---------|
| CrossArms   | 0.8469 | 0.8781 | 0.8735 | 0.8720 | 0.8840 | 0.9136  | 0.9175  | 0.9152  | 0.9033  | 0.9260  |
| DrinkGlas   | 0.8659 | 0.8687 | 0.8457 | 0.8536 | 0.8745 | 0.9127  | 0.9198  | 0.9150  | 0.9175  | 0.9121  |
| Entrainment | 0.8904 | 0.8407 | 0.6774 | 0.6278 | 0.8819 | 0.9580  | 0.9624  | 0.9591  | 0.9632  | 0.9576  |
| HoldWeight  | 0.8881 | 0.9106 | 0.9175 | 0.8486 | 0.9306 | 0.9175  | 0.9175  | 0.9127  | 0.9056  | 0.9191  |
| LiftHold    | 0.9008 | 0.9062 | 0.9085 | 0.9058 | 0.8748 | 0.9129  | 0.9157  | 0.9129  | 0.9150  | 0.9171  |
| PointFinger | 0.7521 | 0.8265 | 0.8193 | 0.7686 | 0.8623 | 0.8763  | 0.9106  | 0.9008  | 0.9128  | 0.8985  |
| Relaxed     | 0.9333 | 0.9400 | 0.9390 | 0.9287 | 0.9375 | 0.9602  | 0.9602  | 0.9591  | 0.9470  | 0.9470  |
| StretchHold | 0.8964 | 0.8710 | 0.9034 | 0.8635 | 0.9072 | 0.9152  | 0.9152  | 0.9178  | 0.9099  | 0.9191  |
| TouchIndex  | 0.8056 | 0.8385 | 0.8168 | 0.8780 | 0.8583 | 0.8691  | 0.8918  | 0.9080  | 0.8480  | 0.8865  |
| TouchNose   | 0.6798 | 0.7111 | 0.7454 | 0.7349 | 0.7173 | 0.8239  | 0.8267  | 0.8698  | 0.8130  | 0.8057  |

### Key Observations
- **LSTM dominates CNN** on all tasks (avg 0.9098 vs 0.8501 combined).
- **Entrainment** is the best task for LSTM (0.9601), worst for CNN (0.7836) — likely task-specific signal that LSTMs capture well.
- **Relaxed** is CNN's best task (0.9357) and second-best for LSTM (0.9547).
- **TouchNose** is hardest for both models (CNN: 0.7177, LSTM: 0.8278).
- CNN Entrainment has very high variance (std=0.1094) due to folds 3-4 collapsing.

---

## 8. Nested CV Transformer (Optuna Hyperparameter Search)

> 5-outer-fold nested CV with 3-inner-fold Optuna search (20 trials, 20 epochs per trial), final training for 80 epochs.
> Model: Transformer encoder; tasks: HC vs PD and PD vs DD (separate binary classifiers).

### Per-Fold Test Accuracy

| Fold | HC vs PD Acc | PD vs DD Acc | Combined Acc |
|------|-------------|-------------|-------------|
| 1    | 0.9359      | 0.9251      | 0.9305      |
| 2    | 0.9355      | 0.9259      | 0.9307      |
| 3    | 0.9355      | 0.9251      | 0.9303      |
| 4    | 0.9357      | 0.9271      | 0.9314      |
| 5    | 0.9378      | 0.9251      | 0.9315      |
| **Mean ± Std** | **0.9361 ± 0.0009** | **0.9257 ± 0.0007** | **0.9309 ± 0.0005** |

### Best Hyperparameters per Fold

| Fold | model_dim | num_heads | num_layers | d_ff | dropout | lr       |
|------|-----------|-----------|------------|------|---------|----------|
| 1    | 32        | 4         | 2          | 128  | 0.159   | 6.78e-4  |
| 2    | 32        | 4         | 2          | 128  | 0.119   | 9.94e-4  |
| 3    | 32        | 4         | 3          | 128  | 0.111   | 5.02e-4  |
| 4    | 64        | 4         | 2          | 128  | 0.118   | 6.54e-4  |
| 5    | 64        | 8         | 2          | 256  | 0.118   | 4.66e-4  |

> Very stable results across folds (std < 0.001). Consistently favors small models (dim=32–64, 2 layers) with low dropout (~0.11–0.16).

---

## Summary Table

| Model | Task | HC vs PD Acc | PD vs DD Acc | Combined Acc |
|-------|------|-------------|-------------|-------------|
| BaseModel + BandPass | Binary | **0.9312 ± 0.0043** | 0.8704 ± 0.0366 | 0.9008 ± 0.0205 |
| BaseModel (no BandPass) | Binary | 0.8598 ± 0.0274 | 0.8539 ± 0.0233 | 0.8569 ± 0.0254 |
| ThreeClass BaseModel | 3-class | — | — | 0.8737 ± 0.0165 |
| SSL Full Fine-tune (100%) | Binary (3-fold avg) | 0.9356 | 0.9250 | 0.9303 |
| SSL Full Fine-tune (20%) | Binary (3-fold avg) | 0.9226 | 0.9167 | 0.9196 |
| SSL Linear Probe (100%) | Binary | **0.9412** | **0.9390** | **0.9401** |
| Nested CV Transformer | Binary | 0.9361 ± 0.0009 | 0.9257 ± 0.0007 | 0.9309 ± 0.0005 |
| TimesFM LoRA (4-fold) | Binary | 0.9362 ± 0.0006 | 0.9289 ± 0.0039 | 0.9326 ± 0.0022 |
| CNN Task-Wise (Relaxed) | Binary (best task) | 0.9582 | 0.9132 | 0.9357 |
| LSTM Task-Wise (Entrainment) | Binary (best task) | 0.9643 | 0.9558 | **0.9601** |
| CNN Task-Wise (avg all tasks) | Binary | 0.8721 | 0.8282 | 0.8501 |
| LSTM Task-Wise (avg all tasks) | Binary | 0.9280 | 0.8917 | 0.9098 |

### Key Observations
1. **Band-pass filtering is critical**: adds ~7 pp on HC vs PD (0.9312 vs 0.8598) and ~1.6 pp on PD vs DD.
2. **PD is the easiest class**: three-class model achieves 99.89% PD accuracy; HC is the hardest (77.86%).
3. **SSL full fine-tune is competitive at 93%** with 100% labels, and already ~92% at just 20% of labels.
4. **SSL linear probe now matches full fine-tune** (~94.0%) after the updated run — frozen features are sufficient given enough samples and longer training.
5. **Nested CV Transformer is the most consistent** overall model (std < 0.001 across 5 folds).
6. **LSTM on single-task (Entrainment) is strongest** (96.01%), but task selection matters — TouchNose drops to 82.78%.
7. **CNN is highly task-dependent**: Relaxed=93.6%, but TouchNose=71.8% and Entrainment=78.4%.
8. **TimesFM LoRA is competitive across 4 folds** (HC vs PD 0.9362 ± 0.0006, PD vs DD 0.9289 ± 0.0039) — matches the nested-CV transformer on HC vs PD and slightly edges it on PD vs DD.
