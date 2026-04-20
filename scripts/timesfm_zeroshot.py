# =============================================================================
# TimesFM Zero-Shot Evaluation Pipeline
# =============================================================================
# Mirrors the LoRA fine-tuning pipeline exactly (same dataset, same k-fold
# split, same metrics format) but replaces training with frozen feature
# extraction + lightweight probes (k-NN and Logistic Regression).
#
# Purpose: Show that without fine-tuning, the foundation model does NOT
# generalise well to this domain, establishing a lower-bound baseline.
# =============================================================================

# !pip install "timesfm[torch] @ git+https://github.com/google-research/timesfm.git"

import pathlib
import numpy as np
import json
import torch
import torch.nn as nn
import warnings
from scipy.signal import butter, filtfilt, resample_poly
from math import gcd
import os
import csv
from tqdm import tqdm
from collections import defaultdict

from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, roc_curve, auc
)
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')


# =============================================================================
# SIGNAL PROCESSING  (identical to fine-tuning pipeline)
# =============================================================================

def create_windows(data, window_size=256, overlap=0):
    n_samples, n_channels = data.shape
    step = int(window_size * (1 - overlap))
    windows = []
    for start in range(0, n_samples - window_size + 1, step):
        windows.append(data[start:start + window_size, :])
    return np.array(windows) if windows else None


def downsample(data, original_freq=100, target_freq=64):
    g = gcd(original_freq, target_freq)
    return resample_poly(data, target_freq // g, original_freq // g, axis=0)


def bandpass_filter(signal, original_freq=64, upper_bound=20, lower_bound=0.1):
    nyquist = 0.5 * original_freq
    b, a = butter(5, [lower_bound / nyquist, upper_bound / nyquist], btype='band')
    return filtfilt(b, a, signal, axis=0)


# =============================================================================
# DATASET  (identical field layout to LoRA pipeline)
# =============================================================================

class ParkinsonsDataLoader(torch.utils.data.Dataset):
    TASKS = ["CrossArms", "DrinkGlas", "Entrainment", "HoldWeight", "LiftHold",
             "PointFinger", "Relaxed", "StretchHold", "TouchIndex", "TouchNose"]

    def __init__(self, data_root=None, window_size=256,
                 left_samples=None, right_samples=None,
                 hc_vs_pd=None, pd_vs_dd=None,
                 apply_dowsampling=True,
                 apply_bandpass_filter=True,
                 **kwargs):

        self.apply_dowsampling    = apply_dowsampling
        self.apply_bandpass_filter = apply_bandpass_filter
        self.data_root            = data_root

        if data_root is not None:
            self.window_size         = window_size
            self.patients_template   = pathlib.Path(data_root) / "patients" / "patient_{p:03d}.json"
            self.timeseries_template = (pathlib.Path(data_root) / "movement" / "timeseries"
                                        / "{N:03d}_{X}_{Y}.txt")
            self.left_samples  = []
            self.right_samples = []
            self.hc_vs_pd      = []
            self.pd_vs_dd      = []
            self.patient_ids   = []
            self.task_names    = []
            self._load_data()
        else:
            self.left_samples  = np.array(left_samples)  if left_samples  is not None else np.array([])
            self.right_samples = np.array(right_samples) if right_samples is not None else np.array([])
            self.hc_vs_pd      = np.array(hc_vs_pd)      if hc_vs_pd      is not None else np.array([])
            self.pd_vs_dd      = np.array(pd_vs_dd)      if pd_vs_dd      is not None else np.array([])
            self.patient_ids   = np.array(kwargs.get('patient_ids', []))
            self.task_names    = np.array(kwargs.get('task_names',  []))

    def _load_data(self):
        for patient_id in tqdm(range(1, 470), desc="Loading patients"):
            p_path = pathlib.Path(str(self.patients_template).format(p=patient_id))
            if not p_path.exists():
                continue
            try:
                with open(p_path) as f:
                    condition = json.load(f).get('condition', '')

                if condition == 'Healthy':
                    hc_lbl, pd_lbl, overlap = 0, -1, 0.70
                elif 'Parkinson' in condition:
                    hc_lbl, pd_lbl, overlap = 1,  0, 0.0
                else:
                    hc_lbl, pd_lbl, overlap = -1, 1, 0.65

                for task in self.TASKS:
                    lp = pathlib.Path(str(self.timeseries_template).format(
                        N=patient_id, X=task, Y="LeftWrist"))
                    rp = pathlib.Path(str(self.timeseries_template).format(
                        N=patient_id, X=task, Y="RightWrist"))
                    if not (lp.exists() and rp.exists()):
                        continue
                    try:
                        ld = np.loadtxt(lp, delimiter=",")
                        rd = np.loadtxt(rp, delimiter=",")
                        ld = ld[:, :6] if ld.shape[1] > 6 else ld
                        ld = ld[50:, :] if ld.shape[0] > 50 else ld
                        rd = rd[:, :6] if rd.shape[1] > 6 else rd
                        rd = rd[50:, :] if rd.shape[0] > 50 else rd
                        if self.apply_dowsampling:
                            ld = downsample(ld); rd = downsample(rd)
                        if self.apply_bandpass_filter:
                            ld = bandpass_filter(ld); rd = bandpass_filter(rd)
                        lw = create_windows(ld, self.window_size, overlap)
                        rw = create_windows(rd, self.window_size, overlap)
                        if lw is not None and rw is not None:
                            n = min(len(lw), len(rw))
                            for i in range(n):
                                self.left_samples.append(lw[i])
                                self.right_samples.append(rw[i])
                                self.hc_vs_pd.append(hc_lbl)
                                self.pd_vs_dd.append(pd_lbl)
                                self.patient_ids.append(patient_id)
                                self.task_names.append(task)
                    except Exception:
                        continue
            except Exception:
                continue

        self.left_samples  = np.array(self.left_samples)
        self.right_samples = np.array(self.right_samples)
        self.hc_vs_pd      = np.array(self.hc_vs_pd)
        self.pd_vs_dd      = np.array(self.pd_vs_dd)
        self.patient_ids   = np.array(self.patient_ids)
        self.task_names    = np.array(self.task_names)

    def __len__(self):
        return len(self.left_samples)

    def __getitem__(self, idx):
        return (
            torch.FloatTensor(self.left_samples[idx]),
            torch.FloatTensor(self.right_samples[idx]),
            torch.LongTensor([self.hc_vs_pd[idx]]).squeeze(),
            torch.LongTensor([self.pd_vs_dd[idx]]).squeeze(),
        )


# =============================================================================
# PATIENT-LEVEL K-FOLD SPLIT  (identical to LoRA pipeline)
# =============================================================================

def build_patient_folds(data_root, k=5):
    """Stratified K-Fold at the patient level — guarantees no patient leakage."""
    tmpl = pathlib.Path(data_root) / "patients" / "patient_{p:03d}.json"
    cond_map = {}
    for pid in range(1, 470):
        p = pathlib.Path(str(tmpl).format(p=pid))
        if p.exists():
            try:
                with open(p) as f:
                    c = json.load(f).get('condition', '')
                cond_map[pid] = 0 if c == 'Healthy' else (1 if 'Parkinson' in c else 2)
            except Exception:
                pass

    pids   = sorted(cond_map)
    labels = [cond_map[p] for p in pids]

    hc = labels.count(0); pd_ = labels.count(1); dd = labels.count(2)
    print(f"Total patients: {len(pids)}  (HC={hc}, PD={pd_}, DD={dd})")

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    return pids, labels, list(skf.split(pids, labels))


def make_subset(full_ds, patient_ids):
    mask = np.isin(full_ds.patient_ids, list(patient_ids))
    return ParkinsonsDataLoader(
        data_root=None,
        left_samples=full_ds.left_samples[mask],
        right_samples=full_ds.right_samples[mask],
        hc_vs_pd=full_ds.hc_vs_pd[mask],
        pd_vs_dd=full_ds.pd_vs_dd[mask],
        patient_ids=full_ds.patient_ids[mask],
        task_names=full_ds.task_names[mask],
    )


# =============================================================================
# FROZEN TIMESFM FEATURE EXTRACTOR
# =============================================================================

class FrozenTimesFMExtractor(nn.Module):
    """
    Loads TimesFM 2.5-200M, freezes ALL parameters, and extracts
    mean-pooled patch embeddings for both wrists (12 channels total).

    Output shape: (batch, 1280) — channel-mean of all 12 sensor channels.
    Zero trainable parameters — true zero-shot evaluation.
    """

    def __init__(self):
        super().__init__()
        try:
            import timesfm
            self.tfm = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
                "google/timesfm-2.5-200m-pytorch"
            )
            # Register inner nn.Module so .to(device) propagates
            if hasattr(self.tfm, 'model') and isinstance(self.tfm.model, nn.Module):
                self.register_module('_inner', self.tfm.model)
            # Freeze every single parameter
            for param in self.parameters():
                param.requires_grad = False
            print("✓ TimesFM 2.5-200M loaded and fully frozen (zero-shot mode)")
        except Exception as e:
            raise RuntimeError(f"TimesFM failed to load: {e}")

    @torch.no_grad()
    def forward(self, left_wrist: torch.Tensor, right_wrist: torch.Tensor) -> torch.Tensor:
        """
        Args:
            left_wrist  : (B, seq_len, 6)
            right_wrist : (B, seq_len, 6)
        Returns:
            features    : (B, 1280)
        """
        device = left_wrist.device
        B, seq_len, _ = left_wrist.shape
        patch_len  = 32
        n_channels = 12

        # Concatenate both wrists → (B, T, 12) → (B*12, T)
        combined = torch.cat([left_wrist, right_wrist], dim=-1)
        flat     = combined.permute(0, 2, 1).reshape(B * n_channels, seq_len)

        # Pad to multiple of patch_len
        if seq_len % patch_len != 0:
            pad  = patch_len - (seq_len % patch_len)
            flat = torch.cat([flat, torch.zeros(B * n_channels, pad, device=device)], dim=1)

        num_patches = flat.shape[1] // patch_len
        patched     = flat.view(B * n_channels, num_patches, patch_len)
        mask        = torch.zeros_like(patched)      # 0 = observed

        model = self.tfm.model if hasattr(self.tfm, 'model') else self.tfm
        results, _ = model(patched, mask)
        embeddings  = results[1]                     # (B*12, num_patches, 1280)

        pooled      = embeddings.mean(dim=1)         # (B*12, 1280)
        per_channel = pooled.view(B, n_channels, -1) # (B, 12, 1280)
        features    = per_channel.mean(dim=1)        # (B, 1280)

        return features


# =============================================================================
# FEATURE EXTRACTION LOOP
# =============================================================================

def extract_all_features(extractor, dataset, batch_size=32, num_workers=0, device='cpu'):
    """Run frozen extractor over a dataset, return (features, hc_labels, pd_labels)."""
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    all_feats, all_hc, all_pd = [], [], []

    extractor.eval()
    with torch.no_grad():
        for left, right, hc_pd, pd_dd in tqdm(loader, desc="  Extracting", leave=False):
            left, right = left.to(device), right.to(device)
            feats = extractor(left, right)
            all_feats.append(feats.cpu().numpy())
            all_hc.append(hc_pd.numpy())
            all_pd.append(pd_dd.numpy())

    return (np.vstack(all_feats),
            np.concatenate(all_hc),
            np.concatenate(all_pd))


# =============================================================================
# METRICS  (same format as LoRA pipeline)
# =============================================================================

def calculate_metrics(y_true, y_pred, task_name="", verbose=True):
    if len(y_true) == 0:
        return {}

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    accuracy                      = accuracy_score(y_true, y_pred)
    prec, rec, f1, sup            = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0)
    prec_avg, rec_avg, f1_avg, _  = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    metrics = {
        'accuracy':           round(float(accuracy),  4),
        'precision_avg':      round(float(prec_avg),  4),
        'recall_avg':         round(float(rec_avg),   4),
        'f1_avg':             round(float(f1_avg),    4),
        'precision_per_class': prec,
        'recall_per_class':    rec,
        'f1_per_class':        f1,
        'support_per_class':   sup,
        'confusion_matrix':    cm,
    }

    if verbose and task_name:
        print(f"\n  === {task_name} ===")
        print(f"  Accuracy : {accuracy:.4f}")
        print(f"  Precision: {prec_avg:.4f}  Recall: {rec_avg:.4f}  F1: {f1_avg:.4f}")
        unique = np.unique(y_true)
        for i, lbl in enumerate(unique):
            if i < len(prec):
                name = ("HC" if lbl == 0 else "PD") if "HC" in task_name else ("PD" if lbl == 0 else "DD")
                print(f"    {name}: P={prec[i]:.4f}  R={rec[i]:.4f}  F1={f1[i]:.4f}  n={sup[i]}")
        print(f"  Confusion Matrix:\n{cm}")

    return metrics


def _flat_metrics_row(method, task, fold, metrics):
    """Flatten metrics dict into a single CSV row."""
    row = {
        'fold':          fold,
        'method':        method,
        'task':          task,
        'n_samples':     int(metrics.get('support_per_class', np.array([])).sum())
                         if len(metrics.get('support_per_class', [])) else 0,
        'accuracy':      metrics.get('accuracy',      0),
        'precision_avg': metrics.get('precision_avg', 0),
        'recall_avg':    metrics.get('recall_avg',    0),
        'f1_avg':        metrics.get('f1_avg',        0),
    }
    # Per-class breakdown
    for i, (p, r, f, s) in enumerate(zip(
        metrics.get('precision_per_class', []),
        metrics.get('recall_per_class',    []),
        metrics.get('f1_per_class',        []),
        metrics.get('support_per_class',   []),
    )):
        tag = f'cls{i}'
        row[f'precision_{tag}'] = round(float(p), 4)
        row[f'recall_{tag}']    = round(float(r), 4)
        row[f'f1_{tag}']        = round(float(f), 4)
        row[f'support_{tag}']   = int(s)

    # TP / FP / TN / FN from confusion matrix
    cm = metrics.get('confusion_matrix', None)
    if cm is not None:
        for i in range(cm.shape[0]):
            tp = int(cm[i, i])
            fp = int(cm[:, i].sum() - tp)
            fn = int(cm[i, :].sum() - tp)
            tn = int(cm.sum() - tp - fp - fn)
            tag = f'cls{i}'
            row[f'TP_{tag}'] = tp; row[f'FP_{tag}'] = fp
            row[f'TN_{tag}'] = tn; row[f'FN_{tag}'] = fn

    return row


# =============================================================================
# ZERO-SHOT PROBES
# =============================================================================

def run_knn(X_tr, y_tr, X_vl, y_vl, k_values, task, fold, scaler, verbose=True):
    rows = []
    X_tr_s = scaler.transform(X_tr)
    X_vl_s = scaler.transform(X_vl)
    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k, metric='cosine', n_jobs=-1)
        knn.fit(X_tr_s, y_tr)
        preds = knn.predict(X_vl_s)
        m = calculate_metrics(y_vl, preds, f"k-NN k={k} | {task}", verbose=verbose)
        rows.append(_flat_metrics_row(f'knn_k{k}', task, fold, m))
    return rows


def run_logreg(X_tr, y_tr, X_vl, y_vl, task, fold, scaler, max_iter=2000, verbose=True):
    X_tr_s = scaler.transform(X_tr)
    X_vl_s = scaler.transform(X_vl)
    clf = LogisticRegression(C=1.0, max_iter=max_iter, solver='lbfgs',
                              multi_class='auto', random_state=42, n_jobs=-1)
    clf.fit(X_tr_s, y_tr)
    preds = clf.predict(X_vl_s)
    probs = clf.predict_proba(X_vl_s)
    m = calculate_metrics(y_vl, preds, f"LogReg | {task}", verbose=verbose)
    row = _flat_metrics_row('logreg', task, fold, m)
    return row, preds, probs


# =============================================================================
# VISUALISATIONS  (mirrors LoRA pipeline plots)
# =============================================================================

def plot_tsne(features, hc_labels, pd_labels, fold_idx, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    print("  Running t-SNE …")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    f2d  = tsne.fit_transform(features)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # HC vs PD
    mask = hc_labels != -1
    if mask.any():
        ax = axes[0]
        for lbl, col, name in [(0, 'royalblue', 'HC'), (1, 'tomato', 'PD')]:
            m = mask & (hc_labels == lbl)
            if m.any():
                ax.scatter(f2d[m, 0], f2d[m, 1], c=col, alpha=0.5, s=18,
                           linewidths=0, label=f'{name} (n={m.sum()})')
        ax.set_title('Frozen TimesFM — HC vs PD')
        ax.set_xlabel('t-SNE 1'); ax.set_ylabel('t-SNE 2')
        ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # PD vs DD
    mask = pd_labels != -1
    if mask.any():
        ax = axes[1]
        for lbl, col, name in [(0, 'seagreen', 'PD'), (1, 'darkorange', 'DD')]:
            m = mask & (pd_labels == lbl)
            if m.any():
                ax.scatter(f2d[m, 0], f2d[m, 1], c=col, alpha=0.5, s=18,
                           linewidths=0, label=f'{name} (n={m.sum()})')
        ax.set_title('Frozen TimesFM — PD vs DD')
        ax.set_xlabel('t-SNE 1'); ax.set_ylabel('t-SNE 2')
        ax.legend(fontsize=9); ax.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, f'zeroshot_tsne_fold{fold_idx+1}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ t-SNE → {path}")


def plot_roc(y_true, y_probs, task, fold_idx, output_dir):
    """ROC curve for the binary classification tasks."""
    if len(np.unique(y_true)) < 2:
        return
    os.makedirs(output_dir, exist_ok=True)
    # Use column 1 (positive class probability)
    pos_probs = y_probs[:, 1] if y_probs.ndim == 2 else y_probs
    fpr, tpr, _ = roc_curve(y_true, pos_probs)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, lw=2, label=f'ROC (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
    plt.title(f'Zero-Shot ROC — {task} | Fold {fold_idx+1}')
    plt.legend(); plt.grid(alpha=0.3)
    path = os.path.join(output_dir, f'roc_{task.replace(" ", "_")}_fold{fold_idx+1}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_summary_bar(summary_rows, output_dir):
    """Bar chart comparing methods across tasks."""
    os.makedirs(output_dir, exist_ok=True)
    methods = list(dict.fromkeys(r['method'] for r in summary_rows))
    tasks   = list(dict.fromkeys(r['task']   for r in summary_rows))

    fig, axes = plt.subplots(1, len(tasks), figsize=(6 * len(tasks), 5))
    if len(tasks) == 1:
        axes = [axes]

    for ax, task in zip(axes, tasks):
        acc_means = []
        acc_stds  = []
        for m in methods:
            subset = [r for r in summary_rows if r['method'] == m and r['task'] == task]
            if subset:
                acc_means.append(subset[0].get('accuracy_mean', 0))
                acc_stds.append(subset[0].get('accuracy_std',  0))
            else:
                acc_means.append(0); acc_stds.append(0)

        x = np.arange(len(methods))
        bars = ax.bar(x, acc_means, yerr=acc_stds, capsize=5, width=0.6,
                      color=['#4C72B0', '#DD8452', '#55A868', '#C44E52'][:len(methods)],
                      alpha=0.85)
        ax.set_xticks(x); ax.set_xticklabels(methods, rotation=15, ha='right')
        ax.set_ylim(0, 1.05)
        ax.set_ylabel('Accuracy (mean ± std)')
        ax.set_title(f'Zero-Shot — {task}')
        ax.axhline(0.5, color='red', linestyle='--', lw=1, alpha=0.5, label='Random')
        ax.legend(fontsize=8); ax.grid(axis='y', alpha=0.3)
        for bar, mean in zip(bars, acc_means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f'{mean:.3f}', ha='center', va='bottom', fontsize=8)

    plt.suptitle('TimesFM Zero-Shot Baseline (Frozen Backbone)', fontsize=13, y=1.01)
    plt.tight_layout()
    path = os.path.join(output_dir, 'zeroshot_summary_bar.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Summary bar chart → {path}")


# =============================================================================
# CSV UTILITIES
# =============================================================================

def save_csv(rows, path):
    if not rows:
        return
    all_keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction='ignore')
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, '') for k in all_keys})
    print(f"✓ Saved → {path}")


def build_summary(all_rows, methods, tasks):
    """Aggregate per-fold rows into mean ± std per method × task."""
    summary = []
    for method in methods:
        for task in tasks:
            subset = [r for r in all_rows
                      if r.get('method') == method and r.get('task') == task]
            if not subset:
                continue
            row = {'method': method, 'task': task, 'n_folds': len(subset)}
            for key in ['accuracy', 'precision_avg', 'recall_avg', 'f1_avg']:
                vals = [r[key] for r in subset if key in r]
                if vals:
                    row[f'{key}_mean'] = round(float(np.mean(vals)), 4)
                    row[f'{key}_std']  = round(float(np.std(vals)),  4)
            summary.append(row)
    return summary


# =============================================================================
# MAIN
# =============================================================================

def main():
    config = {
        # ── Dataset ────────────────────────────────────────────────────────
        'data_root': (
            "/kaggle/input/datasets/meherujannat/parkinsons/"
            "pads-parkinsons-disease-smartwatch-dataset-1.0.0"
        ),
        'apply_downsampling':    True,
        'apply_bandpass_filter': True,
        'window_size':           256,

        # ── Evaluation ─────────────────────────────────────────────────────
        'num_folds':   5,
        'batch_size':  64,      # no gradients → large batches are fine
        'num_workers': 0,
        'knn_k_values':    [5, 15, 25],
        'logreg_max_iter': 2000,

        # ── Output ─────────────────────────────────────────────────────────
        'output_dir':   'results/timesfm_zeroshot',
        'create_tsne':  True,
        'create_plots': True,
    }

    metrics_dir = os.path.join(config['output_dir'], 'metrics')
    plots_dir   = os.path.join(config['output_dir'], 'plots')
    ckpt_dir    = os.path.join(config['output_dir'], 'checkpoints')   # for config.json parity
    for d in [metrics_dir, plots_dir, ckpt_dir]:
        os.makedirs(d, exist_ok=True)

    # Persist config (mirrors LoRA pipeline)
    cfg_serialisable = {k: v for k, v in config.items()
                        if isinstance(v, (str, int, float, bool, list, dict, type(None)))}
    cfg_serialisable['ablation_type'] = 'zeroshot'
    with open(os.path.join(ckpt_dir, 'config.json'), 'w') as f:
        json.dump(cfg_serialisable, f, indent=2)
    print(f"✓ Config saved: {ckpt_dir}/config.json")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # ── Load TimesFM once (shared across all folds) ───────────────────────
    print("\nLoading TimesFM (zero-shot / fully frozen) …")
    extractor = FrozenTimesFMExtractor().to(device)
    total_params     = sum(p.numel() for p in extractor.parameters())
    trainable_params = sum(p.numel() for p in extractor.parameters() if p.requires_grad)
    print(f"  Total params    : {total_params:,}")
    print(f"  Trainable params: {trainable_params:,}  ← must be 0 for zero-shot")
    assert trainable_params == 0, "ERROR: extractor has trainable params — zero-shot violated!"

    # ── Load full dataset once ────────────────────────────────────────────
    print("\nLoading dataset …")
    full_ds = ParkinsonsDataLoader(
        data_root=config['data_root'],
        window_size=config['window_size'],
        apply_dowsampling=config['apply_downsampling'],
        apply_bandpass_filter=config['apply_bandpass_filter'],
    )
    print(f"Total windows loaded: {len(full_ds)}")

    # ── Patient-level k-fold splits (identical to LoRA pipeline) ─────────
    pids, plabels, fold_splits = build_patient_folds(config['data_root'], k=config['num_folds'])

    all_rows       = []   # per-fold metric rows
    all_methods_set = set()

    for fold_idx, (train_idx, val_idx) in enumerate(fold_splits):
        print(f"\n{'='*65}")
        print(f"  FOLD {fold_idx + 1} / {config['num_folds']}  (zero-shot)")
        print(f"{'='*65}")

        train_pids = {pids[i] for i in train_idx}
        val_pids   = {pids[i] for i in val_idx}

        train_ds = make_subset(full_ds, train_pids)
        val_ds   = make_subset(full_ds, val_pids)

        # Report class distribution — mirrors LoRA pipeline printout
        tr_hc = int((train_ds.hc_vs_pd == 0).sum())
        tr_pd = int(((train_ds.hc_vs_pd == 1) & (train_ds.pd_vs_dd == 0)).sum())
        tr_dd = int((train_ds.pd_vs_dd  == 1).sum())
        va_hc = int((val_ds.hc_vs_pd   == 0).sum())
        va_pd = int(((val_ds.hc_vs_pd  == 1) & (val_ds.pd_vs_dd == 0)).sum())
        va_dd = int((val_ds.pd_vs_dd   == 1).sum())
        print(f"  Train: {len(train_ds)} windows  (HC={tr_hc}, PD={tr_pd}, DD={tr_dd})")
        print(f"  Val  : {len(val_ds)}  windows  (HC={va_hc}, PD={va_pd}, DD={va_dd})")

        # ── Extract frozen features (no gradients, no backbone updates) ───
        print("\n  Extracting train features …")
        X_train, y_hc_train, y_pd_train = extract_all_features(
            extractor, train_ds, config['batch_size'], config['num_workers'], device)

        print("  Extracting val features …")
        X_val, y_hc_val, y_pd_val = extract_all_features(
            extractor, val_ds, config['batch_size'], config['num_workers'], device)

        print(f"  Feature shapes: train={X_train.shape}  val={X_val.shape}")

        # Fit scaler on TRAIN features only — val set never touches .fit()
        scaler = StandardScaler()
        scaler.fit(X_train)

        # t-SNE of frozen val features
        if config['create_tsne']:
            plot_tsne(X_val, y_hc_val, y_pd_val, fold_idx, plots_dir)

        # Build valid-sample masks (exclude -1 sentinel)
        hc_tr_m = y_hc_train != -1;  hc_vl_m = y_hc_val != -1
        pd_tr_m = y_pd_train != -1;  pd_vl_m = y_pd_val != -1

        print(f"\n  HC vs PD — train={hc_tr_m.sum()}  val={hc_vl_m.sum()}")
        print(f"  PD vs DD — train={pd_tr_m.sum()}  val={pd_vl_m.sum()}")

        fold_plot_dir = os.path.join(plots_dir, f'fold_{fold_idx+1}')
        os.makedirs(fold_plot_dir, exist_ok=True)

        # ── k-NN probes ───────────────────────────────────────────────────
        print("\n  --- k-NN (zero-shot probe) ---")
        knn_hc_rows = run_knn(
            X_train[hc_tr_m], y_hc_train[hc_tr_m],
            X_val[hc_vl_m],   y_hc_val[hc_vl_m],
            config['knn_k_values'], 'HC_vs_PD', fold_idx + 1, scaler
        )
        knn_pd_rows = run_knn(
            X_train[pd_tr_m], y_pd_train[pd_tr_m],
            X_val[pd_vl_m],   y_pd_val[pd_vl_m],
            config['knn_k_values'], 'PD_vs_DD', fold_idx + 1, scaler
        )
        for r in knn_hc_rows + knn_pd_rows:
            all_rows.append(r)
            all_methods_set.add(r['method'])

        # ── Logistic Regression probes ────────────────────────────────────
        print("\n  --- Logistic Regression (zero-shot linear probe) ---")
        lr_hc_row, lr_hc_preds, lr_hc_probs = run_logreg(
            X_train[hc_tr_m], y_hc_train[hc_tr_m],
            X_val[hc_vl_m],   y_hc_val[hc_vl_m],
            'HC_vs_PD', fold_idx + 1, scaler, config['logreg_max_iter']
        )
        lr_pd_row, lr_pd_preds, lr_pd_probs = run_logreg(
            X_train[pd_tr_m], y_pd_train[pd_tr_m],
            X_val[pd_vl_m],   y_pd_val[pd_vl_m],
            'PD_vs_DD', fold_idx + 1, scaler, config['logreg_max_iter']
        )
        for r in [lr_hc_row, lr_pd_row]:
            all_rows.append(r)
            all_methods_set.add(r['method'])

        # ROC curves for LogReg (mirrors LoRA pipeline)
        if config['create_plots']:
            if hc_vl_m.sum() > 0:
                plot_roc(y_hc_val[hc_vl_m], lr_hc_probs, 'HC_vs_PD', fold_idx, fold_plot_dir)
            if pd_vl_m.sum() > 0:
                plot_roc(y_pd_val[pd_vl_m], lr_pd_probs, 'PD_vs_DD', fold_idx, fold_plot_dir)

        # Save per-fold metrics immediately (robust to interruptions)
        save_csv(all_rows, os.path.join(metrics_dir, 'zeroshot_per_fold.csv'))

    # ── Build and save summary CSV ────────────────────────────────────────
    all_methods = [f'knn_k{k}' for k in config['knn_k_values']] + ['logreg']
    all_tasks   = ['HC_vs_PD', 'PD_vs_DD']
    summary     = build_summary(all_rows, all_methods, all_tasks)
    save_csv(summary, os.path.join(metrics_dir, 'zeroshot_summary.csv'))

    # Summary bar chart
    if config['create_plots']:
        plot_summary_bar(summary, plots_dir)

    # ── Print final table (same format as LoRA pipeline) ──────────────────
    print(f"\n{'='*70}")
    print("TIMESFM ZERO-SHOT BASELINE  (mean ± std across folds)")
    print(f"{'='*70}")
    print(f"{'Method':<14} {'Task':<12} {'Accuracy':>12} {'Precision':>13} {'Recall':>11} {'F1':>11}")
    print("-" * 70)
    for r in summary:
        print(f"{r['method']:<14} {r['task']:<12} "
              f"{r.get('accuracy_mean',0):.4f}±{r.get('accuracy_std',0):.4f}  "
              f"{r.get('precision_avg_mean',0):.4f}±{r.get('precision_avg_std',0):.4f}  "
              f"{r.get('recall_avg_mean',0):.4f}±{r.get('recall_avg_std',0):.4f}  "
              f"{r.get('f1_avg_mean',0):.4f}±{r.get('f1_avg_std',0):.4f}")
    print(f"{'='*70}")
    print("\nNote: Sub-random or near-random scores confirm that frozen TimesFM")
    print("features do NOT transfer zero-shot to this wrist-tremor domain,")
    print("motivating the LoRA / gradual-unfreeze fine-tuning experiments.")

    return all_rows, summary


if __name__ == "__main__":
    main()