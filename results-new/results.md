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

| % Labels | N Samples | HC vs PD Acc | PD vs DD Acc | Combined Acc | HC F1   | PD F1   |
|----------|-----------|--------------|--------------|--------------|---------|---------|
| 5%       | 747       | 0.8052       | 0.8046       | 0.8049       | 0.8044  | 0.8022  |
| 10%      | 1,494     | 0.8762       | 0.8612       | 0.8687       | 0.8745  | 0.8602  |
| 20%      | 2,987     | 0.9222       | 0.8982       | 0.9102       | 0.9216  | 0.8977  |
| 50%      | 7,466     | 0.9270       | 0.9169       | 0.9220       | 0.9264  | 0.9165  |
| 70%      | 10,453    | 0.9234       | 0.9210       | 0.9222       | 0.9227  | 0.9204  |
| 100%     | 14,932    | 0.9335       | 0.9240       | 0.9287       | 0.9327  | 0.9234  |

> Reaches ~91% combined accuracy with only 20% of labels.

---

## 5. SSL BaseModel — Linear Probe (Label Efficiency)

| % Labels | N Samples | HC vs PD Acc | PD vs DD Acc | Combined Acc | HC F1   | PD F1   |
|----------|-----------|--------------|--------------|--------------|---------|---------|
| 5%       | 747       | 0.7298       | 0.6658       | 0.6978       | 0.7288  | 0.6658  |
| 10%      | 1,494     | 0.7060       | 0.6497       | 0.6779       | 0.7054  | 0.6498  |
| 20%      | 2,987     | 0.7286       | 0.7414       | 0.7350       | 0.7265  | 0.7370  |
| 50%      | 7,466     | 0.8218       | 0.7695       | 0.7956       | 0.8178  | 0.7663  |
| 70%      | 10,453    | 0.8190       | 0.7979       | 0.8084       | 0.8170  | 0.7954  |
| 100%     | 14,932    | 0.8310       | 0.7953       | 0.8132       | 0.8268  | 0.7912  |

> Linear probe tops out at ~81% — significant gap vs full fine-tune, confirming the SSL representations need adaptation.

---

## 6. TimesFM LoRA (Fold 1 Only)

| Task     | Best Epoch | Accuracy | Precision | Recall | F1     |
|----------|------------|----------|-----------|--------|--------|
| HC vs PD | 16         | 0.9331   | 0.9346    | 0.9331 | 0.9327 |
| PD vs DD | 10         | 0.9266   | 0.9349    | 0.9266 | 0.9261 |

> Only 1 fold available. Results are promising but incomplete.

---

## 7. CNN-LSTM Task-Wise

> No results yet — output files are empty (training not completed).

---

## Summary Table

| Model | Task | Accuracy (Mean ± Std) | F1 (Mean ± Std) |
|-------|------|-----------------------|-----------------|
| BaseModel + BandPass | HC vs PD | **0.9312 ± 0.0043** | 0.9302 ± 0.0046 |
| BaseModel + BandPass | PD vs DD | 0.8704 ± 0.0366 | 0.8701 ± 0.0365 |
| BaseModel (no BandPass) | HC vs PD | 0.8598 ± 0.0274 | 0.8596 ± 0.0275 |
| BaseModel (no BandPass) | PD vs DD | 0.8539 ± 0.0233 | 0.8533 ± 0.0238 |
| ThreeClass BaseModel | HC/PD/DD | 0.8737 ± 0.0165 | — |
| SSL Full Fine-tune (100%) | HC vs PD | 0.9335 | 0.9327 |
| SSL Full Fine-tune (100%) | PD vs DD | 0.9240 | 0.9234 |
| SSL Linear Probe (100%) | HC vs PD | 0.8310 | 0.8268 |
| SSL Linear Probe (100%) | PD vs DD | 0.7953 | 0.7912 |
| TimesFM LoRA (fold 1) | HC vs PD | 0.9331 | 0.9327 |
| TimesFM LoRA (fold 1) | PD vs DD | 0.9266 | 0.9261 |

### Key Observations
1. **Band-pass filtering is critical**: adds ~7 pp accuracy on HC vs PD (0.9312 vs 0.8598) and ~1.6 pp on PD vs DD.
2. **PD is the easiest class to detect**: three-class model achieves 99.89% PD accuracy; HC is the hardest (77.86%).
3. **SSL full fine-tune is competitive**: reaches 0.9287 combined accuracy at 100% labels, and already ~0.91 at just 20%.
4. **SSL linear probe lags significantly**: maxes at 0.8132 — the frozen SSL features alone are insufficient.
5. **TimesFM LoRA looks strong** on fold 1 (≥0.926 on both tasks) but needs full 5-fold evaluation to confirm.
