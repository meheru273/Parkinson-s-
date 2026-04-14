import pathlib
import numpy as np
import json
import copy
import shutil
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from typing import Dict, List, Tuple
import warnings
from scipy.signal import butter, filtfilt, resample_poly
from math import gcd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.manifold import TSNE
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from collections import defaultdict
import os
import math
import csv
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import joblib
warnings.filterwarnings('ignore')

# ============================================================================
#  Checkpoint Helpers
# ============================================================================
CHECKPOINT_DIR = "checkpoints"
RUN_STATE_FILE  = "checkpoints/run_state.json"

# Previous Kaggle run outputs (read-only input mount).
# The script will fall back to this directory when local resume files are absent.
PREV_RUN_DIR = ""  # Set from config['resume_from'] at runtime


def save_checkpoint(outer_fold_idx, epoch, model, optimizer, scheduler, scaler,
                    history, best_test_acc, best_model_state, patience_counter,
                    fold_metrics_hc, fold_metrics_pd,
                    train_metrics_history_hc, train_metrics_history_pd,
                    best_probs_dict):
    """Atomically save mid-training state for one outer fold."""
    path = os.path.join(CHECKPOINT_DIR, f"fold_{outer_fold_idx+1}_mid_training.pth")
    tmp  = path + ".tmp"
    torch.save({
        'outer_fold_idx':           outer_fold_idx,
        'epoch':                    epoch,
        'model_state_dict':         model.state_dict(),
        'optimizer_state_dict':     optimizer.state_dict(),
        'scheduler_state_dict':     scheduler.state_dict(),
        'scaler_state_dict':        scaler.state_dict(),
        'history':                  dict(history),
        'best_test_acc':            best_test_acc,
        'best_model_state':         best_model_state,
        'patience_counter':         patience_counter,
        'fold_metrics_hc':          fold_metrics_hc,
        'fold_metrics_pd':          fold_metrics_pd,
        'train_metrics_history_hc': train_metrics_history_hc,
        'train_metrics_history_pd': train_metrics_history_pd,
        'best_probs_dict':          best_probs_dict,
    }, tmp)
    os.replace(tmp, path)   # atomic rename – safe if Kaggle is killed mid-write
    print(f"  [ckpt] fold {outer_fold_idx+1} epoch {epoch+1} saved → {path}")


def load_checkpoint(outer_fold_idx, model, optimizer, scheduler, scaler):
    """Load mid-training checkpoint if it exists.
    Returns (start_epoch, history, best_test_acc, best_model_state,
             patience_counter, fold_metrics_hc, fold_metrics_pd,
             train_metrics_history_hc, train_metrics_history_pd, best_probs_dict)
    or None if no checkpoint exists.
    Falls back to PREV_RUN_DIR when no local checkpoint is present.
    """
    fname = f"fold_{outer_fold_idx+1}_mid_training.pth"
    path = os.path.join(CHECKPOINT_DIR, fname)
    if not os.path.exists(path):
        # Fall back to previous Kaggle run's output
        prev_path = os.path.join(PREV_RUN_DIR, "checkpoints", fname)
        if os.path.exists(prev_path):
            print(f"  [ckpt] No local checkpoint – copying from previous run: {prev_path}")
            shutil.copy2(prev_path, path)
        else:
            return None
    print(f"  [ckpt] Resuming fold {outer_fold_idx+1} from {path}")
    ck = torch.load(path, map_location='cpu')
    model.load_state_dict(ck['model_state_dict'])
    optimizer.load_state_dict(ck['optimizer_state_dict'])
    scheduler.load_state_dict(ck['scheduler_state_dict'])
    scaler.load_state_dict(ck['scaler_state_dict'])
    history = defaultdict(list, ck['history'])
    return (
        ck['epoch'] + 1,          # resume from the NEXT epoch
        history,
        ck['best_test_acc'],
        ck['best_model_state'],
        ck['patience_counter'],
        ck['fold_metrics_hc'],
        ck['fold_metrics_pd'],
        ck['train_metrics_history_hc'],
        ck['train_metrics_history_pd'],
        ck['best_probs_dict'],
    )


def save_run_state(outer_fold_idx, all_outer_results, best_hyperparams_per_fold):
    """Save global run state so outer folds can be skipped on restart."""
    def _jsonify(obj):
        if isinstance(obj, dict):
            return {k: _jsonify(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_jsonify(v) for v in obj]
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, defaultdict):
            return dict(obj)
        return obj
    state = {
        'completed_outer_folds': outer_fold_idx + 1,
        'best_hyperparams_per_fold': _jsonify(best_hyperparams_per_fold),
        'all_outer_results': _jsonify([
            {k: v for k, v in r.items() if k != 'history'}
            for r in all_outer_results
        ]),
    }
    with open(RUN_STATE_FILE, 'w') as f:
        json.dump(state, f, indent=2)


def load_run_state():
    """Return (completed_folds, best_hyperparams_per_fold, all_outer_results)
    or (0, [], []) if no run state exists.
    Falls back to PREV_RUN_DIR when no local run_state.json is present.
    """
    # Prefer local (current-run) state; fall back to previous run's state
    candidate = RUN_STATE_FILE
    if not os.path.exists(candidate):
        prev_candidate = os.path.join(PREV_RUN_DIR, "checkpoints/run_state.json")
        if os.path.exists(prev_candidate):
            candidate = prev_candidate
            print(f"  [resume] No local run_state.json found – loading from previous run: {candidate}")
        else:
            return 0, [], []
    with open(candidate, 'r') as f:
        s = json.load(f)
    print(f"  [resume] Found run_state.json: {s['completed_outer_folds']} outer folds already done.")
    return s['completed_outer_folds'], s['best_hyperparams_per_fold'], s['all_outer_results']


def fold_is_done(outer_fold_idx):
    # Check local working directory first
    local_done = os.path.join(CHECKPOINT_DIR, f"fold_{outer_fold_idx+1}.done")
    if os.path.exists(local_done):
        return True
    # Fall back to previous Kaggle run's output
    prev_done = os.path.join(PREV_RUN_DIR, "checkpoints", f"fold_{outer_fold_idx+1}.done")
    return os.path.exists(prev_done)


def mark_fold_done(outer_fold_idx):
    open(os.path.join(CHECKPOINT_DIR, f"fold_{outer_fold_idx+1}.done"), 'w').close()
    # Remove mid-training checkpoint (no longer needed)
    mid_ck = os.path.join(CHECKPOINT_DIR, f"fold_{outer_fold_idx+1}_mid_training.pth")
    if os.path.exists(mid_ck):
        os.remove(mid_ck)


def bootstrap_from_prev_run(prev_run_dir):
    """Copy resumable artifacts from a previous Kaggle run's output into
    the current working directories so that all resume functions find them
    locally.  Copies:
      - checkpoints/*.done, *_mid_training.pth, run_state.json, config.json,
        best_model_*.pth
      - optuna_studies/*.db  (SQLite Optuna storage)
    Already-existing local files are NEVER overwritten (local wins).
    """
    if not prev_run_dir or not os.path.isdir(prev_run_dir):
        return

    def _copy_dir(src_subdir, dst_dir, glob_pattern="*"):
        import glob
        src = os.path.join(prev_run_dir, src_subdir)
        if not os.path.isdir(src):
            return
        os.makedirs(dst_dir, exist_ok=True)
        for filepath in glob.glob(os.path.join(src, glob_pattern)):
            if not os.path.isfile(filepath):
                continue
            dst_path = os.path.join(dst_dir, os.path.basename(filepath))
            if os.path.exists(dst_path):
                continue  # never overwrite local files
            print(f"  [bootstrap] {filepath} → {dst_path}")
            shutil.copy2(filepath, dst_path)

    _copy_dir("checkpoints", CHECKPOINT_DIR)          # .done, mid_training, run_state, best_model, config
    _copy_dir("optuna_studies", "optuna_studies", "*.db")  # Optuna SQLite DBs
    print("  [bootstrap] Previous-run artifacts copied.")


def create_windows(data, window_size=256, overlap=0):
    
    n_samples, n_channels = data.shape
    step = int(window_size * (1 - overlap))   
    windows = []
    for start in range(0, n_samples - window_size + 1, step):
        end = start + window_size
        windows.append(data[start:end, :])
    
    return np.array(windows) if windows else None


#down sampling 
def downsample(data, original_freq=100, target_freq=64):
    g = gcd(original_freq, target_freq)
    up = target_freq // g
    down = original_freq // g
    return resample_poly(data, up, down, axis=0)


# band pass filter
def bandpass_filter(signal, original_freq=64, upper_bound=20, lower_bound=0.1):
    nyquist = 0.5 * original_freq
    low = lower_bound / nyquist
    high = upper_bound / nyquist
    b, a = butter(5, [low, high], btype='band')
    return filtfilt(b, a, signal, axis=0)

# ============================================================================
#  Splitting Methods
# ============================================================================
def k_fold_split_method(data_root, full_dataset, k=5):
   
    patient_conditions = {}
    patients_template = pathlib.Path(data_root) / "patients" / "patient_{p:03d}.json"
    
    dataset_patient_ids = set(full_dataset.patient_ids)
    
    for patient_id in dataset_patient_ids:
        patient_path = pathlib.Path(str(patients_template).format(p=patient_id))
        if patient_path.exists():
            try:
                with open(patient_path, 'r') as f:
                    condition = json.load(f).get('condition', 'Unknown')
                    patient_conditions[patient_id] = condition
            except:
                pass
            
    patient_list = []
    patient_labels = []
    for pid in sorted(patient_conditions.keys()):
        condition = patient_conditions[pid]
        if condition == 'Healthy':
            label = 0
        elif 'Parkinson' in condition:
            label = 1
        else:
            label = 2
        patient_list.append(pid)
        patient_labels.append(label)
    
    print(f"Total patients: {len(patient_list)} (HC={patient_labels.count(0)}, PD={patient_labels.count(1)}, DD={patient_labels.count(2)})")
    
    if len(patient_list) == 0:
        raise ValueError("No patients found! Check if data_root path is correct and patient files exist.")
    
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    fold_datasets = []
    
    for fold_id, (train_idx, test_idx) in enumerate(skf.split(patient_list, patient_labels)):
        train_patients = set([patient_list[i] for i in train_idx])
        test_patients = set([patient_list[i] for i in test_idx])
        
        train_mask = np.array([pid in train_patients for pid in full_dataset.patient_ids])
        test_mask = np.array([pid in test_patients for pid in full_dataset.patient_ids])
        
        train_dataset = type(full_dataset)(
            data_root=None,
            left_samples=full_dataset.left_samples[train_mask],
            right_samples=full_dataset.right_samples[train_mask],
            hc_vs_pd=full_dataset.hc_vs_pd[train_mask],
            pd_vs_dd=full_dataset.pd_vs_dd[train_mask],
            patient_ids=full_dataset.patient_ids[train_mask]
        )
        
        test_dataset = type(full_dataset)(
            data_root=None,
            left_samples=full_dataset.left_samples[test_mask],
            right_samples=full_dataset.right_samples[test_mask],
            hc_vs_pd=full_dataset.hc_vs_pd[test_mask],
            pd_vs_dd=full_dataset.pd_vs_dd[test_mask],
            patient_ids=full_dataset.patient_ids[test_mask]
        )
        
        # Print fold info
        train_hc = np.sum(train_dataset.hc_vs_pd == 0)
        train_pd = np.sum((train_dataset.hc_vs_pd == 1) & (train_dataset.pd_vs_dd == 0))
        train_dd = np.sum(train_dataset.pd_vs_dd == 1)
        test_hc = np.sum(test_dataset.hc_vs_pd == 0)
        test_pd = np.sum((test_dataset.hc_vs_pd == 1) & (test_dataset.pd_vs_dd == 0))
        test_dd = np.sum(test_dataset.pd_vs_dd == 1)
        
        print(f"\nFold {fold_id+1}/{k}:")
        print(f"  Train: {len(train_dataset)} samples (HC={train_hc}, PD={train_pd}, DD={train_dd})")
        print(f"  Test:  {len(test_dataset)} samples (HC={test_hc}, PD={test_pd}, DD={test_dd})")
        
        fold_datasets.append((train_dataset, test_dataset))
    
    return fold_datasets
# ============================================================================
#  DataLoader
# ============================================================================
class ParkinsonsDataLoader(Dataset):
    
    def __init__(self, data_root: str = None, window_size: int = 256, 
                 left_samples=None, right_samples=None, 
                 hc_vs_pd=None, pd_vs_dd=None,
                 apply_dowsampling=True,
                 apply_bandpass_filter=True, **kwargs):
        
        self.left_samples = []
        self.right_samples = []
        self.hc_vs_pd = []
        self.pd_vs_dd = []
        self.patient_ids = []  
        self.task_names = []   
        self.apply_dowsampling = apply_dowsampling
        self.apply_bandpass_filter = apply_bandpass_filter
        self.data_root = data_root

        if data_root is not None:
            self.window_size = window_size
            self.patients_template = pathlib.Path(data_root) / "patients" / "patient_{p:03d}.json"
            self.timeseries_template = pathlib.Path(data_root) / "movement" / "timeseries" / "{N:03d}_{X}_{Y}.txt"
            self.questionnaires_template = pathlib.Path(data_root) / "questionnaire" / "questionnaire_response_{p:03d}.json"
            
            # Tasks 
            self.tasks = ["CrossArms", "DrinkGlas", "Entrainment", "HoldWeight", "LiftHold", 
                         "PointFinger", "Relaxed", "StretchHold", "TouchIndex", "TouchNose"]
            self.wrists = ["LeftWrist", "RightWrist"]
            
            self.patient_ids_list = list(range(1, 470))
            print(f"Dataset: {len(self.patient_ids_list)} patients (001-469)")
        
            self._load_data()
        else:
            if left_samples is not None:
                self.left_samples = np.array(left_samples) if not isinstance(left_samples, np.ndarray) else left_samples
            if right_samples is not None:
                self.right_samples = np.array(right_samples) if not isinstance(right_samples, np.ndarray) else right_samples
            if hc_vs_pd is not None:
                self.hc_vs_pd = np.array(hc_vs_pd) if not isinstance(hc_vs_pd, np.ndarray) else hc_vs_pd
            if pd_vs_dd is not None:
                self.pd_vs_dd = np.array(pd_vs_dd) if not isinstance(pd_vs_dd, np.ndarray) else pd_vs_dd
            
            self.patient_ids = kwargs.get('patient_ids', [])
            if self.patient_ids is not None and len(self.patient_ids) > 0:
                self.patient_ids = np.array(self.patient_ids) if not isinstance(self.patient_ids, np.ndarray) else self.patient_ids


    def _load_data(self):
        for patient_id in tqdm(self.patient_ids_list, desc="Loading patients"):
            patient_path = pathlib.Path(str(self.patients_template).format(p=patient_id))
            questionnaire_path = pathlib.Path(str(self.questionnaires_template).format(p=patient_id))
            
            if not patient_path.exists():
                continue
                
            try:
                with open(patient_path, 'r') as f:
                    metadata = json.load(f)
                
                condition = metadata.get('condition', '')
            
                if condition == 'Healthy':
                    hc_vs_pd_label = 0  # Healthy
                    pd_vs_dd_label = -1  # Not applicable for PD vs DD 
                    overlap = 0.70      #overlap for differential sampling 
                elif 'Parkinson' in condition:
                    hc_vs_pd_label = 1  
                    pd_vs_dd_label = 0   # Parkinson's for PD vs DD
                    overlap = 0           #overlap for differential sampling
                else:  
                    hc_vs_pd_label = -1  # Not applicable for HC vs PD 
                    pd_vs_dd_label = 1   # Other disorders
                    overlap = 0.65       #overlap for differential sampling

                patient_left_samples = []
                patient_right_samples = []
                patient_task_names = []
                
                for task in self.tasks:
                    left_path = pathlib.Path(str(self.timeseries_template).format(
                        N=patient_id, X=task, Y="LeftWrist"))
                    right_path = pathlib.Path(str(self.timeseries_template).format(
                        N=patient_id, X=task, Y="RightWrist"))
                    
                    if not (left_path.exists() and right_path.exists()):
                        continue
                        
                    try:
                        left_data = np.loadtxt(left_path, delimiter=",")
                        right_data = np.loadtxt(right_path, delimiter=",")
                        
                        if left_data.shape[1] > 6:
                            left_data = left_data[:, :6]  # Take first 6 channels
                        if left_data.shape[0] > 50:
                            left_data = left_data[50:, :]  # Skip first 0.5 sec
                        
                        if right_data.shape[1] > 6:
                            right_data = right_data[:, :6]
                        if right_data.shape[0] > 50:
                            right_data = right_data[50:, :]
                        
                        # Downsample 
                        if self.apply_dowsampling:
                            left_data = downsample(left_data)
                            right_data = downsample(right_data)
                            
                        if self.apply_bandpass_filter:
                            left_data = bandpass_filter(left_data)
                            right_data = bandpass_filter(right_data)

                        if left_data is None or right_data is None:
                            continue
                        
                        # Create windows
                        left_windows = create_windows(left_data, self.window_size, overlap=overlap)
                        right_windows = create_windows(right_data, self.window_size, overlap=overlap)

                        if left_windows is not None and right_windows is not None:
                            min_windows = min(len(left_windows), len(right_windows))
                            
                            for i in range(min_windows):
                                patient_left_samples.append(left_windows[i])
                                patient_right_samples.append(right_windows[i])
                                patient_task_names.append(task)
                        
                    except Exception as e:
                        print(f"Error loading data for patient {patient_id}, task {task}: {e}")
                        continue
                
                if len(patient_left_samples) > 0:
                    n_samples = len(patient_left_samples)
                    
                    for i in range(n_samples):
                        self.left_samples.append(patient_left_samples[i])
                        self.right_samples.append(patient_right_samples[i])
                        self.hc_vs_pd.append(hc_vs_pd_label)
                        self.pd_vs_dd.append(pd_vs_dd_label)
                        self.patient_ids.append(patient_id)
                        self.task_names.append(patient_task_names[i])
                
            except Exception as e:
                print(f"Error loading patient {patient_id}: {e}")
                continue
        
        self.left_samples = np.array(self.left_samples)
        self.right_samples = np.array(self.right_samples)
        self.hc_vs_pd = np.array(self.hc_vs_pd)
        self.pd_vs_dd = np.array(self.pd_vs_dd)
        self.patient_ids = np.array(self.patient_ids)
        self.task_names = np.array(self.task_names)


    def get_train_test_split(self, split_type=1, **kwargs):
            
        if split_type == 3:
            k = kwargs.get('k', 10)
            
            if self.data_root is None:
                raise ValueError("data_root is required for K-fold split")
            
            fold_datasets = k_fold_split_method(self.data_root, self, k)
            
            return fold_datasets
        
        else:
            raise ValueError(f"Invalid split_type: {split_type}. Use 1 (patient-level), 2 (task-based), or 3 (k-fold)")


    def __len__(self):
        return len(self.left_samples) if hasattr(self, 'left_samples') and isinstance(self.left_samples, (list, np.ndarray)) else 0
    
    def __getitem__(self, idx):
        left_sample = torch.FloatTensor(self.left_samples[idx])
        right_sample = torch.FloatTensor(self.right_samples[idx])
        hc_vs_pd = torch.LongTensor([self.hc_vs_pd[idx]])
        pd_vs_dd = torch.LongTensor([self.pd_vs_dd[idx]])
        
        return left_sample, right_sample, hc_vs_pd.squeeze(), pd_vs_dd.squeeze()
    
    
# ============================================================================
#  Main Model 
# ============================================================================
class PositionalEncoding(nn.Module):
    def __init__(self, model_dim: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, model_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, model_dim, 2).float() * (-math.log(10000.0) / model_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)  
    
    def forward(self, x):
        timestep = x.size(1)
        return x + self.pe[:timestep, :].unsqueeze(0) 


class FeedForward(nn.Module):
    def __init__(self, model_dim: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(model_dim, d_ff)
        self.linear2 = nn.Linear(d_ff, model_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, x):
        residual = x
        x = self.linear1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.layer_norm(x + residual)
        return x


class CrossAttention(nn.Module):
    def __init__(self, model_dim: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        
        self.cross_attention_1to2 = nn.MultiheadAttention(embed_dim=model_dim,num_heads=num_heads,dropout=dropout,batch_first=True)
        self.cross_attention_2to1 = nn.MultiheadAttention(embed_dim=model_dim,num_heads=num_heads,dropout=dropout,batch_first=True)
        
        self.self_attention_1 = nn.MultiheadAttention(embed_dim=model_dim,num_heads=num_heads,dropout=dropout,batch_first=True)
        self.self_attention_2 = nn.MultiheadAttention(embed_dim=model_dim,num_heads=num_heads,dropout=dropout,batch_first=True)
        
        
        # Layer norms for residual connections
        self.norm_cross_1 = nn.LayerNorm(model_dim)
        self.norm_cross_2 = nn.LayerNorm(model_dim)
        self.norm_self_1 = nn.LayerNorm(model_dim)
        self.norm_self_2 = nn.LayerNorm(model_dim)
        
        self.feed_forward_1 = FeedForward(model_dim, d_ff, dropout)       
        self.feed_forward_2 = FeedForward(model_dim, d_ff, dropout)
        
    def forward(self, channel_1, channel_2):
        # Cross attention with residual connections
        channel_1_cross_attn, _ = self.cross_attention_1to2(query=channel_1,key=channel_2,value=channel_2)
        channel_1_cross = self.norm_cross_1(channel_1 + channel_1_cross_attn)
        
        channel_2_cross_attn, _ = self.cross_attention_2to1(query=channel_2,key=channel_1,value=channel_1)
        channel_2_cross = self.norm_cross_2(channel_2 + channel_2_cross_attn)
        
        # Self attention with residual connections
        channel_1_self_attn, _ = self.self_attention_1(query=channel_1_cross,key=channel_1_cross,value=channel_1_cross)
        channel_1_self = self.norm_self_1(channel_1_cross + channel_1_self_attn)
        
        channel_2_self_attn, _ = self.self_attention_2(query=channel_2_cross,key=channel_2_cross,value=channel_2_cross)
        channel_2_self = self.norm_self_2(channel_2_cross + channel_2_self_attn)
        
        # Feed forward
        channel_1_out = self.feed_forward_1(channel_1_self)
        channel_2_out = self.feed_forward_2(channel_2_self)

        return channel_1_out, channel_2_out


class MainModel(nn.Module):
    def __init__(
        self,
        input_dim: int = 6,  
        model_dim: int = 128,
        num_heads: int = 8,
        num_layers: int = 4,
        d_ff: int = 512,
        dropout: float = 0.1,
        timestep: int = 256,
        num_classes: int = 2,
        fusion_method: str = 'concat',
    ):
        super().__init__()
        
        self.model_dim = model_dim
        self.timestep = timestep
        self.fusion_method = fusion_method
        
        self.left_projection = nn.Linear(input_dim, model_dim)
        self.right_projection = nn.Linear(input_dim, model_dim)
        
        self.positional_encoding = PositionalEncoding(model_dim, max_len=timestep)
        
        self.layers = nn.ModuleList([
            CrossAttention(model_dim, num_heads, d_ff, dropout) 
            for _ in range(num_layers)
        ])
            
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        fusion_dim = model_dim * 2

        # Classification heads
        self.head_hc_vs_pd = nn.Sequential(
            nn.Linear(fusion_dim, model_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim, 2)  # Binary: HC vs PD
        )
        
        self.head_pd_vs_dd = nn.Sequential(
            nn.Linear(fusion_dim, model_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim, 2)  # Binary: PD vs DD
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def get_features(self, left_wrist, right_wrist, device=None):

        left_encoded = self.left_projection(left_wrist)   
        right_encoded = self.right_projection(right_wrist) 

        left_encoded = self.positional_encoding(left_encoded)
        right_encoded = self.positional_encoding(right_encoded)
        
        left_encoded = self.dropout(left_encoded)
        right_encoded = self.dropout(right_encoded)
        
        for layer in self.layers:
            left_encoded, right_encoded = layer(left_encoded, right_encoded)

        left_pool = self.global_pool(left_encoded.transpose(1, 2)).squeeze(-1)
        right_pool = self.global_pool(right_encoded.transpose(1, 2)).squeeze(-1)

        fused_signal_features = torch.cat([left_pool, right_pool], dim=1)  
        
        fused_features = fused_signal_features
            
        return fused_features
        
    def forward(self, left_wrist, right_wrist, device=None):
        
        left_encoded = self.left_projection(left_wrist)   
        right_encoded = self.right_projection(right_wrist) 

        left_encoded = self.positional_encoding(left_encoded)
        right_encoded = self.positional_encoding(right_encoded)
        
        left_encoded = self.dropout(left_encoded)
        right_encoded = self.dropout(right_encoded)
        
        for layer in self.layers:
            left_encoded, right_encoded = layer(left_encoded, right_encoded)

        left_pool = self.global_pool(left_encoded.transpose(1, 2)).squeeze(-1)
        right_pool = self.global_pool(right_encoded.transpose(1, 2)).squeeze(-1)

        fused_signal_features = torch.cat([left_pool, right_pool], dim=1)  
        
        fused_features = fused_signal_features

        logits_hc_vs_pd = self.head_hc_vs_pd(fused_features)
        logits_pd_vs_dd = self.head_pd_vs_dd(fused_features)

        return logits_hc_vs_pd, logits_pd_vs_dd

# ============================================================================
#  Evaluation Functions and Plots
# ============================================================================
def calculate_metrics(y_true, y_pred, task_name="", verbose=True):
    if len(y_true) == 0:
        return {}
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
    precision_avg, recall_avg, f1_avg, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
    
    cm = confusion_matrix(y_true, y_pred)
    
    metrics = {
        'accuracy': accuracy,
        'precision_per_class': precision,
        'recall_per_class': recall,
        'f1_per_class': f1,
        'support_per_class': support,
        'precision_avg': precision_avg,
        'recall_avg': recall_avg,
        'f1_avg': f1_avg,
        'confusion_matrix': cm
    }
    
    if verbose and task_name:
        print(f"\n=== {task_name} Metrics ===")
        print(f"Accuracy: {accuracy:.4f}")
        print(f" Precision: {precision_avg:.4f}")
        print(f" Recall: {recall_avg:.4f}")
        print(f"F1: {f1_avg:.4f}")
        
        unique_labels = np.unique(np.concatenate([y_true, y_pred]))
        for i, label in enumerate(unique_labels):
            if i < len(precision):
                label_name = "HC" if label == 0 else ("PD" if label == 1 else f"Class_{label}")
                if task_name == "PD vs DD":
                    label_name = "PD" if label == 0 else ("DD" if label == 1 else f"Class_{label}")
                print(f"{label_name}: Precision={precision[i]:.4f}, Recall={recall[i]:.4f}, F1={f1[i]:.4f}, Support={support[i]}")
        
        print("Confusion Matrix:")
        print(cm)
    
    return metrics

def save_fold_metric(fold_idx, fold_suffix,
                     fold_metrics_hc, fold_metrics_pd,
                     train_metrics_history_hc, train_metrics_history_pd,
                     train_losses, val_losses):
    """Save comprehensive per-classifier metrics for each epoch.
    
    For EACH classifier (hc_vs_pd and pd_vs_dd), saves:
      - Train & val/test: accuracy, per-class precision/recall/F1/support
      - Weighted-average precision/recall/F1
      - Confusion matrix (flattened)
      - Train loss, val/test loss
    """
    os.makedirs("metrics", exist_ok=True)

    def write_csv(filename, val_metrics_list, train_metrics_list, t_losses, v_losses):
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                "epoch", "train_loss", "val_loss",
                # Train metrics
                "train_accuracy",
                "train_precision_avg", "train_recall_avg", "train_f1_avg",
                "train_precision_per_class", "train_recall_per_class",
                "train_f1_per_class", "train_support_per_class",
                "train_confusion_matrix",
                # Val/test metrics
                "val_accuracy",
                "val_precision_avg", "val_recall_avg", "val_f1_avg",
                "val_precision_per_class", "val_recall_per_class",
                "val_f1_per_class", "val_support_per_class",
                "val_confusion_matrix",
            ])
            n_epochs = max(len(val_metrics_list), len(train_metrics_list))
            for i in range(n_epochs):
                epoch_num = i + 1
                tl = t_losses[i] if i < len(t_losses) else ''
                vl = v_losses[i] if i < len(v_losses) else ''

                # Train metrics for this epoch
                tm = train_metrics_list[i] if i < len(train_metrics_list) else {}
                # Val metrics for this epoch
                vm = val_metrics_list[i]['metrics'] if i < len(val_metrics_list) else {}

                def fmt_arr(arr):
                    if arr is None: return ''
                    return '|'.join(f'{x:.6f}' if isinstance(x, float) else str(x) for x in np.asarray(arr).flatten())

                writer.writerow([
                    epoch_num, tl, vl,
                    # Train
                    tm.get('accuracy', ''),
                    tm.get('precision_avg', ''), tm.get('recall_avg', ''), tm.get('f1_avg', ''),
                    fmt_arr(tm.get('precision_per_class')), fmt_arr(tm.get('recall_per_class')),
                    fmt_arr(tm.get('f1_per_class')), fmt_arr(tm.get('support_per_class')),
                    fmt_arr(tm.get('confusion_matrix')),
                    # Val
                    vm.get('accuracy', ''),
                    vm.get('precision_avg', ''), vm.get('recall_avg', ''), vm.get('f1_avg', ''),
                    fmt_arr(vm.get('precision_per_class')), fmt_arr(vm.get('recall_per_class')),
                    fmt_arr(vm.get('f1_per_class')), fmt_arr(vm.get('support_per_class')),
                    fmt_arr(vm.get('confusion_matrix')),
                ])

    # HC vs PD
    if fold_metrics_hc or train_metrics_history_hc:
        hc_filename = f"metrics/hc_vs_pd_metrics{fold_suffix}.csv"
        write_csv(hc_filename, fold_metrics_hc, train_metrics_history_hc, train_losses, val_losses)

    # PD vs DD
    if fold_metrics_pd or train_metrics_history_pd:
        pd_filename = f"metrics/pd_vs_dd_metrics{fold_suffix}.csv"
        write_csv(pd_filename, fold_metrics_pd, train_metrics_history_pd, train_losses, val_losses)



def plot_loss(train_losses, val_losses, output_path):  # FIXED: Changed signature
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss', color='blue')
    plt.plot(val_losses, label='Validation Loss', color='orange')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training and Validation Loss over Epochs', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    

def plot_roc_curves(labels, predictions, probabilities, output_path):
    plt.figure(figsize=(10, 8))
    
    # Calculate ROC curve and AUC
    fpr, tpr, _ = roc_curve(labels, probabilities)
    roc_auc = auc(fpr, tpr)
    
    # Plot
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve', fontsize=14)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_tsne(features, hc_pd_labels, pd_dd_labels, output_dir="plots"):
    
    if features is None or len(features) == 0:
        print("No features available for t-SNE visualization")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    features_2d = tsne.fit_transform(features)

    valid_hc_pd = hc_pd_labels != -1
    valid_pd_dd = pd_dd_labels != -1

    # plot HC vs PD
    if np.any(valid_hc_pd):
        plt.figure(figsize=(8, 6))
        features_hc_pd = features_2d[valid_hc_pd]
        labels_hc_pd = hc_pd_labels[valid_hc_pd]
        
        hc_mask = labels_hc_pd == 0
        pd_mask = labels_hc_pd == 1
        
        if np.any(hc_mask):
            plt.scatter(features_hc_pd[hc_mask,0], features_hc_pd[hc_mask,1], 
                        c='blue', label=f'HC (n={np.sum(hc_mask)})', alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
        if np.any(pd_mask):
            plt.scatter(features_hc_pd[pd_mask,0], features_hc_pd[pd_mask,1], 
                        c='red', label=f'PD (n={np.sum(pd_mask)})', alpha=0.6, s=50, edgecolors='black', linewidth=0.5)

        plt.title("t-SNE: HC vs PD")
        plt.xlabel("t-SNE Component 1"); plt.ylabel("t-SNE Component 2")
        plt.legend(); plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir,"tsne_hc_vs_pd.png"), dpi=150, bbox_inches='tight')
        plt.close()
        print("[saved] tsne_hc_vs_pd.png")

    # plot PD vs DD
    if np.any(valid_pd_dd):
        plt.figure(figsize=(8, 6))
        features_pd_dd = features_2d[valid_pd_dd]
        labels_pd_dd = pd_dd_labels[valid_pd_dd]

        pd_mask = labels_pd_dd == 0
        dd_mask = labels_pd_dd == 1
        
        if np.any(pd_mask):
            plt.scatter(features_pd_dd[pd_mask,0], features_pd_dd[pd_mask,1], 
                        c='green', label=f'PD (n={np.sum(pd_mask)})', alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
        if np.any(dd_mask):
            plt.scatter(features_pd_dd[dd_mask,0], features_pd_dd[dd_mask,1], 
                        c='orange', label=f'DD (n={np.sum(dd_mask)})', alpha=0.6, s=50, edgecolors='black', linewidth=0.5)

        plt.title("t-SNE: PD vs DD")
        plt.xlabel("t-SNE Component 1"); plt.ylabel("t-SNE Component 2")
        plt.legend(); plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir,"tsne_pd_vs_dd.png"), dpi=150, bbox_inches='tight')
        plt.close()

    return features_2d


def save_optuna_results(study, fold_idx, output_dir="optuna_studies"):
    
    os.makedirs(output_dir, exist_ok=True)
    # Save trials to CSV
    import pandas as pd
    trials_df = study.trials_dataframe()
    trials_df.to_csv(f"{output_dir}/fold_{fold_idx}_trials.csv", index=False)
    
# ============================================================================
#  Trainer Functions
# ============================================================================

def train_single_epoch(model, dataloader, criterion_hc, criterion_pd, optimizer, device, scaler):
    """Train for one epoch with AMP mixed-precision."""
    model.train()
    train_loss = 0.0
    hc_pd_train_pred, hc_pd_train_labels = [], []
    pd_dd_train_pred, pd_dd_train_labels = [], []
    use_amp = device.type == 'cuda'
    
    for batch in tqdm(dataloader, desc="Training", leave=False):
        left_sample, right_sample, hc_pd, pd_dd = batch
        
        left_sample = left_sample.to(device, non_blocking=True)
        right_sample = right_sample.to(device, non_blocking=True)
        hc_pd = hc_pd.to(device, non_blocking=True)
        pd_dd = pd_dd.to(device, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)
        
        with autocast(enabled=use_amp):
            hc_pd_logits, pd_dd_logits = model(left_sample, right_sample, device)
            
            total_loss = 0
            loss_count = 0
            
            # HC vs PD loss
            valid_hc_pd_mask = (hc_pd != -1)
            if valid_hc_pd_mask.any():
                valid_logits_hc = hc_pd_logits[valid_hc_pd_mask]
                valid_labels_hc = hc_pd[valid_hc_pd_mask]
                loss_hc = criterion_hc(valid_logits_hc, valid_labels_hc)
                total_loss += loss_hc
                loss_count += 1
                
                preds_hc = torch.argmax(valid_logits_hc, dim=1)
                hc_pd_train_pred.extend(preds_hc.cpu().numpy())
                hc_pd_train_labels.extend(valid_labels_hc.cpu().numpy())
            
            # PD vs DD loss
            valid_pd_dd_mask = (pd_dd != -1)
            if valid_pd_dd_mask.any():
                valid_logits_pd = pd_dd_logits[valid_pd_dd_mask]
                valid_labels_pd = pd_dd[valid_pd_dd_mask]
                loss_pd = criterion_pd(valid_logits_pd, valid_labels_pd)
                total_loss += loss_pd
                loss_count += 1
                
                preds_pd = torch.argmax(valid_logits_pd, dim=1)
                pd_dd_train_pred.extend(preds_pd.cpu().numpy())
                pd_dd_train_labels.extend(valid_labels_pd.cpu().numpy())
        
        # Backward pass with scaler
        if loss_count > 0:
            avg_loss = total_loss / loss_count
            scaler.scale(avg_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            train_loss += avg_loss.item()
    
    train_loss /= len(dataloader)
    
    # Calculate training metrics
    train_metrics_hc = calculate_metrics(hc_pd_train_labels, hc_pd_train_pred,
                                        "Training HC vs PD", verbose=False)
    train_metrics_pd = calculate_metrics(pd_dd_train_labels, pd_dd_train_pred,
                                        "Training PD vs DD", verbose=False)
    
    return train_loss, train_metrics_hc, train_metrics_pd


def validate_single_epoch(model, dataloader, criterion_hc, criterion_pd, device):
    """Validate for one epoch with AMP mixed-precision."""
    model.eval()
    val_loss = 0.0
    hc_pd_val_pred, hc_pd_val_labels, hc_pd_val_probs = [], [], []
    pd_dd_val_pred, pd_dd_val_labels, pd_dd_val_probs = [], [], []
    use_amp = device.type == 'cuda'
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation", leave=False):
            left_sample, right_sample, hc_pd, pd_dd = batch
            
            left_sample = left_sample.to(device, non_blocking=True)
            right_sample = right_sample.to(device, non_blocking=True)
            hc_pd = hc_pd.to(device, non_blocking=True)
            pd_dd = pd_dd.to(device, non_blocking=True)
            
            with autocast(enabled=use_amp):
                hc_pd_logits, pd_dd_logits = model(left_sample, right_sample, device)
                
                total_loss = 0
                loss_count = 0
                
                # HC vs PD loss
                valid_hc_pd_mask = (hc_pd != -1)
                if valid_hc_pd_mask.any():
                    valid_logits_hc = hc_pd_logits[valid_hc_pd_mask]
                    valid_labels_hc = hc_pd[valid_hc_pd_mask]
                    loss_hc = criterion_hc(valid_logits_hc, valid_labels_hc)
                    total_loss += loss_hc
                    loss_count += 1
                    
                    preds_hc = torch.argmax(valid_logits_hc, dim=1)
                    probs_hc = F.softmax(valid_logits_hc.float(), dim=1)[:, 1]
                    hc_pd_val_pred.extend(preds_hc.cpu().numpy())
                    hc_pd_val_labels.extend(valid_labels_hc.cpu().numpy())
                    hc_pd_val_probs.extend(probs_hc.cpu().numpy())
                
                # PD vs DD loss
                valid_pd_dd_mask = (pd_dd != -1)
                if valid_pd_dd_mask.any():
                    valid_logits_pd = pd_dd_logits[valid_pd_dd_mask]
                    valid_labels_pd = pd_dd[valid_pd_dd_mask]
                    loss_pd = criterion_pd(valid_logits_pd, valid_labels_pd)
                    total_loss += loss_pd
                    loss_count += 1
                    
                    preds_pd = torch.argmax(valid_logits_pd, dim=1)
                    probs_pd = F.softmax(valid_logits_pd.float(), dim=1)[:, 1]
                    pd_dd_val_pred.extend(preds_pd.cpu().numpy())
                    pd_dd_val_labels.extend(valid_labels_pd.cpu().numpy())
                    pd_dd_val_probs.extend(probs_pd.cpu().numpy())
            
            if loss_count > 0:
                avg_loss = total_loss / loss_count
                val_loss += avg_loss.item()
    
    val_loss /= len(dataloader)
    
    return (val_loss, hc_pd_val_pred, hc_pd_val_labels, hc_pd_val_probs,
            pd_dd_val_pred, pd_dd_val_labels, pd_dd_val_probs)

def extract_features(model, dataloader, device):
    model.eval()
    all_features = []
    all_hc_pd_labels = []
    all_pd_dd_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting features"):
            left_sample, right_sample, hc_pd, pd_dd = batch
            
            left_sample = left_sample.to(device)
            right_sample = right_sample.to(device)
            
            features = model.get_features(left_sample, right_sample, device) 
            
            all_features.append(features.cpu().numpy())
            all_hc_pd_labels.append(hc_pd.numpy())
            all_pd_dd_labels.append(pd_dd.numpy())
    
    all_features = np.vstack(all_features)
    all_hc_pd_labels = np.concatenate(all_hc_pd_labels)
    all_pd_dd_labels = np.concatenate(all_pd_dd_labels)
    
    return all_features, all_hc_pd_labels, all_pd_dd_labels


# ============================================================================
#  Optuna Objective
# ============================================================================
def objective(trial, train_dataset, val_dataset, base_config, device):
    
    # Sample hyperparameters
    model_dim = trial.suggest_categorical('model_dim', [32, 64])
    num_heads = trial.suggest_categorical('num_heads', [4, 8])
    num_layers = trial.suggest_int('num_layers', 2, 4)
    d_ff = trial.suggest_categorical('d_ff', [128, 256, 512])
    dropout = trial.suggest_float('dropout', 0.1, 0.5)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
    
    num_workers = base_config['num_workers']
    pin = base_config.get('pin_memory', True) and device.type == 'cuda'
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                             num_workers=num_workers, pin_memory=pin,
                             persistent_workers=(num_workers > 0))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                           num_workers=num_workers, pin_memory=pin,
                           persistent_workers=(num_workers > 0))
    
    # Create model
    model = MainModel(
        input_dim=base_config['input_dim'],
        model_dim=model_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        d_ff=d_ff,
        dropout=dropout,
        timestep=base_config['timestep'],
        num_classes=base_config['num_classes']
    ).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                 factor=0.5, patience=5)
    scaler = GradScaler(enabled=(device.type == 'cuda'))
    
    hc_pd_loss = nn.CrossEntropyLoss()
    pd_dd_loss = nn.CrossEntropyLoss()
    
    # Training loop
    best_val_acc = 0.0
    patience_counter = 0
    patience_limit = 10
    
    for epoch in range(base_config.get('optuna_epochs', 30)):
        # Train (suppress per-batch tqdm inside Optuna to keep output clean)
        train_loss, _, _ = train_single_epoch(
            model, train_loader, hc_pd_loss, pd_dd_loss, optimizer, device, scaler
        )
        
        # Validate
        val_results = validate_single_epoch(
            model, val_loader, hc_pd_loss, pd_dd_loss, device
        )
        val_loss, hc_pd_val_pred, hc_pd_val_labels, hc_pd_val_probs, \
        pd_dd_val_pred, pd_dd_val_labels, pd_dd_val_probs = val_results
        
        # Calculate metrics
        val_acc_hc = accuracy_score(hc_pd_val_labels, hc_pd_val_pred) if len(hc_pd_val_labels) > 0 else 0
        val_acc_pd = accuracy_score(pd_dd_val_labels, pd_dd_val_pred) if len(pd_dd_val_labels) > 0 else 0
        val_acc_combined = (val_acc_hc + val_acc_pd) / 2
        
        scheduler.step(val_loss)
        
        # Report intermediate value for pruning
        trial.report(val_acc_combined, epoch)
        
        # Handle pruning
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
        
        # Early stopping
        if val_acc_combined > best_val_acc:
            best_val_acc = val_acc_combined
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience_limit:
                break
    
    return best_val_acc


# ============================================================================
# TRAIN_MODEL FUNCTION
# ============================================================================
def train_model(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Enable cuDNN auto-tuner (speeds up fixed-shape inputs like ours)
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    
    # shared AMP scaler for the final training phase
    scaler = GradScaler(enabled=(device.type == 'cuda'))
    pin = device.type == 'cuda'
    
    # Create output directories
    os.makedirs("metrics", exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs("training_history", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    
    # Save config.json once before training
    config_serializable = {k: v for k, v in config.items()
                           if isinstance(v, (str, int, float, bool, list, dict, type(None)))}
    with open('checkpoints/config.json', 'w') as f:
        json.dump(config_serializable, f, indent=2)
    print("✓ Config saved: checkpoints/config.json")
    
    # Load full dataset
    full_dataset = ParkinsonsDataLoader(
        config['data_root'],
        apply_dowsampling=config['apply_downsampling'],
        apply_bandpass_filter=config['apply_bandpass_filter']
    )
    
    # ====================================================================
    # NESTED CROSS-VALIDATION WITH OPTUNA
    # ====================================================================
    os.makedirs("optuna_studies", exist_ok=True)
    
    # ── Resume: bootstrap artifacts from previous Kaggle run ─────────────────
    global PREV_RUN_DIR
    PREV_RUN_DIR = config.get('resume_from', '')
    bootstrap_from_prev_run(PREV_RUN_DIR)
        
    # Outer CV splits
    outer_folds = full_dataset.get_train_test_split(split_type=3, k=config['outer_folds'])
    
    # ── Resume: load global run state ────────────────────────────────────────
    completed_folds, best_hyperparams_per_fold, all_outer_results = load_run_state()
    
    # OUTER LOOP: Model evaluation
    for outer_fold_idx, (outer_train_dataset, outer_test_dataset) in enumerate(outer_folds):
        print(f"\n{'='*80}")
        print(f"OUTER FOLD {outer_fold_idx + 1}/{config['outer_folds']}")
        print(f"{'='*80}\n")
        
        # Skip already-completed folds (check .done files AND run_state.json)
        if outer_fold_idx < completed_folds or fold_is_done(outer_fold_idx):
            print(f"  [resume] Outer fold {outer_fold_idx+1} already done – skipping.")
            continue
        
        # INNER LOOP: Hyperparameter optimization
        print(f"Starting hyperparameter optimization...")
            
        inner_folds = k_fold_split_method(config['data_root'], outer_train_dataset, 
                                         k=config['inner_folds'])
        
        # Create Optuna study with SQLite storage so it survives interruptions
        storage_url = f"sqlite:///optuna_studies/study_outer_fold_{outer_fold_idx+1}.db"
        study_name  = f"nested_cv_outer_{outer_fold_idx+1}"
        study = optuna.create_study(
            study_name=study_name,
            storage=storage_url,
            load_if_exists=True,   # resumes existing study
            direction='maximize',
            sampler=TPESampler(seed=42),
            pruner=MedianPruner(n_startup_trials=3, n_warmup_steps=5)
        )
        already_done = len([t for t in study.trials
                            if t.state == optuna.trial.TrialState.COMPLETE])
        trials_needed = config['n_trials'] * config['inner_folds']
        print(f"  Optuna study: {already_done}/{trials_needed} trials already done.")
            
        # Optimize on each inner fold
        best_trial_scores = []
        for inner_fold_idx, (inner_train, inner_val) in enumerate(inner_folds):
            # Check how many trials this inner fold needs to run
            inner_trials_done = already_done - inner_fold_idx * config['n_trials']
            remaining = max(0, config['n_trials'] - inner_trials_done)
            if remaining == 0:
                # This inner fold is fully optimised – recover best value from study
                best_trial_scores.append(study.best_value)
                print(f"  Inner Fold {inner_fold_idx+1}: already complete (skipping).")
                continue
            print(f"\n  Inner Fold {inner_fold_idx + 1}/{config['inner_folds']} "
                  f"({remaining}/{config['n_trials']} trials remaining)")
            
            study.optimize(
                lambda trial, tr=inner_train, vl=inner_val: objective(trial, tr, vl, config, device),
                n_trials=remaining,
                show_progress_bar=True
            )
            
            best_trial_scores.append(study.best_value)
            print(f"    Best accuracy: {study.best_value:.4f}")
        
        # Get best hyperparameters
        best_params = study.best_params
        avg_best_score = np.mean(best_trial_scores)
        
        print(f"\n  Best hyperparameters:")
        for param, value in best_params.items():
            print(f"    {param}: {value}")
        print(f"  Average validation accuracy: {avg_best_score:.4f}")        
        best_hyperparams_per_fold.append({
            'outer_fold': outer_fold_idx + 1,
            'params': best_params,
            'avg_val_acc': avg_best_score
        })
        
        # Save Optuna study results (CSV summary)
        save_optuna_results(study, outer_fold_idx + 1)
        # SQLite DB already saved continuously – no joblib dump needed
        
        print(f"\nTraining final model with best hyperparameters...")
        
        train_loader = DataLoader(outer_train_dataset, batch_size=best_params['batch_size'],
                                 shuffle=True, num_workers=config['num_workers'],
                                 pin_memory=pin, persistent_workers=(config['num_workers'] > 0))
        test_loader = DataLoader(outer_test_dataset, batch_size=best_params['batch_size'],
                                shuffle=False, num_workers=config['num_workers'],
                                pin_memory=pin, persistent_workers=(config['num_workers'] > 0))
        
        # Create model with best hyperparameters
        model = MainModel(
            input_dim=config['input_dim'],
            model_dim=best_params['model_dim'],
            num_heads=best_params['num_heads'],
            num_layers=best_params['num_layers'],
            d_ff=best_params['d_ff'],
            dropout=best_params['dropout'],
            timestep=config['timestep'],
            num_classes=config['num_classes']
        ).to(device)
        
        optimizer = optim.AdamW(model.parameters(), 
                               lr=best_params['learning_rate'], 
                               weight_decay=best_params['weight_decay'])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                             factor=0.5, patience=5)
        
        hc_pd_loss = nn.CrossEntropyLoss()
        pd_dd_loss = nn.CrossEntropyLoss()
        
        # ── Attempt to resume from a mid-training checkpoint ──────────────
        ck = load_checkpoint(outer_fold_idx, model, optimizer, scheduler, scaler)
        if ck is not None:
            (start_epoch, history, best_test_acc, best_model_state,
             patience_counter, fold_metrics_hc, fold_metrics_pd,
             train_metrics_history_hc, train_metrics_history_pd,
             best_probs_dict) = ck
            best_hc_pd_probs   = best_probs_dict.get('hc_pd_probs')
            best_hc_pd_preds   = best_probs_dict.get('hc_pd_preds')
            best_hc_pd_labels  = best_probs_dict.get('hc_pd_labels')
            best_pd_dd_probs   = best_probs_dict.get('pd_dd_probs')
            best_pd_dd_preds   = best_probs_dict.get('pd_dd_preds')
            best_pd_dd_labels  = best_probs_dict.get('pd_dd_labels')
            print(f"  Resuming from epoch {start_epoch} "
                  f"(best acc so far: {best_test_acc:.4f}, patience: {patience_counter})")
        else:
            start_epoch             = 0
            history                 = defaultdict(list)
            best_test_acc           = 0.0
            best_model_state        = None
            patience_counter        = 0
            best_hc_pd_probs = best_hc_pd_preds = best_hc_pd_labels = None
            best_pd_dd_probs = best_pd_dd_preds = best_pd_dd_labels = None
            fold_metrics_hc         = []
            fold_metrics_pd         = []
            train_metrics_history_hc = []
            train_metrics_history_pd = []
        
        patience_limit     = config.get('patience', 15)
        ckpt_interval      = config.get('checkpoint_interval', 5)
        
        # Final training loop with early stopping
        for epoch in range(start_epoch, config['final_epochs']):
            print(f"\nOuter Fold {outer_fold_idx + 1}, Epoch {epoch + 1}/{config['final_epochs']}")
            
            train_loss, train_metrics_hc, train_metrics_pd = train_single_epoch(
                model, train_loader, hc_pd_loss, pd_dd_loss, optimizer, device, scaler
            )
            
            test_results = validate_single_epoch(
                model, test_loader, hc_pd_loss, pd_dd_loss, device
            )
            test_loss, hc_pd_test_pred, hc_pd_test_labels, hc_pd_test_probs, \
            pd_dd_test_pred, pd_dd_test_labels, pd_dd_test_probs = test_results
            
            test_acc_hc = accuracy_score(hc_pd_test_labels, hc_pd_test_pred) if len(hc_pd_test_labels) > 0 else 0
            test_acc_pd = accuracy_score(pd_dd_test_labels, pd_dd_test_pred) if len(pd_dd_test_labels) > 0 else 0
            test_acc_combined = (test_acc_hc + test_acc_pd) / 2
            
            train_acc_hc = train_metrics_hc.get('accuracy', 0)
            train_acc_pd = train_metrics_pd.get('accuracy', 0)
            
            scheduler.step(test_loss)
            
            history['train_loss'].append(train_loss)
            history['test_loss'].append(test_loss)
            history['train_acc_hc'].append(train_acc_hc)
            history['train_acc_pd'].append(train_acc_pd)
            history['test_acc_hc'].append(test_acc_hc)
            history['test_acc_pd'].append(test_acc_pd)
            history['test_acc_combined'].append(test_acc_combined)
            
            # Collect per-epoch metrics for comprehensive CSV saving
            train_metrics_history_hc.append(train_metrics_hc)
            train_metrics_history_pd.append(train_metrics_pd)
            
            test_metrics_hc_epoch = calculate_metrics(
                hc_pd_test_labels, hc_pd_test_pred,
                f"Outer Fold {outer_fold_idx + 1} Test HC vs PD", verbose=False
            )
            test_metrics_pd_epoch = calculate_metrics(
                pd_dd_test_labels, pd_dd_test_pred,
                f"Outer Fold {outer_fold_idx + 1} Test PD vs DD", verbose=False
            )
            fold_metrics_hc.append({'epoch': epoch + 1, 'metrics': test_metrics_hc_epoch})
            fold_metrics_pd.append({'epoch': epoch + 1, 'metrics': test_metrics_pd_epoch})
            
            print(f"Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")
            print(f"Test Acc - HC: {test_acc_hc:.4f}, PD: {test_acc_pd:.4f}, Combined: {test_acc_combined:.4f}")
            
            if test_acc_combined > best_test_acc:
                best_test_acc = test_acc_combined
                best_model_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
                
                if hc_pd_test_probs:
                    best_hc_pd_probs = np.array(hc_pd_test_probs)
                    best_hc_pd_preds = np.array(hc_pd_test_pred)
                    best_hc_pd_labels = np.array(hc_pd_test_labels)
                
                if pd_dd_test_probs:
                    best_pd_dd_probs = np.array(pd_dd_test_probs)
                    best_pd_dd_preds = np.array(pd_dd_test_pred)
                    best_pd_dd_labels = np.array(pd_dd_test_labels)
            else:
                patience_counter += 1
                if patience_counter >= patience_limit:
                    print(f"  Early stopping at epoch {epoch+1} (patience={patience_limit})")
                    break
            
            # ── Periodic checkpoint save ────────────────────────────────────
            if (epoch + 1) % ckpt_interval == 0:
                save_checkpoint(
                    outer_fold_idx, epoch, model, optimizer, scheduler, scaler,
                    history, best_test_acc, best_model_state, patience_counter,
                    fold_metrics_hc, fold_metrics_pd,
                    train_metrics_history_hc, train_metrics_history_pd,
                    best_probs_dict={
                        'hc_pd_probs':  best_hc_pd_probs,
                        'hc_pd_preds':  best_hc_pd_preds,
                        'hc_pd_labels': best_hc_pd_labels,
                        'pd_dd_probs':  best_pd_dd_probs,
                        'pd_dd_preds':  best_pd_dd_preds,
                        'pd_dd_labels': best_pd_dd_labels,
                    }
                )
        
        # Save comprehensive metrics CSV for this outer fold
        fold_suffix = f"_nested_cv_fold_{outer_fold_idx + 1}"
        save_fold_metric(
            outer_fold_idx, fold_suffix,
            fold_metrics_hc, fold_metrics_pd,
            train_metrics_history_hc, train_metrics_history_pd,
            history['train_loss'], history['test_loss']
        )
        
        # Load best model
        if best_model_state is None:
            print(f"WARNING: No best model found for outer fold {outer_fold_idx + 1}, using final epoch.")
            best_model_state = copy.deepcopy(model.state_dict())
            best_test_acc = test_acc_combined
        model.load_state_dict(best_model_state)
        
        test_results = validate_single_epoch(model, test_loader, hc_pd_loss, pd_dd_loss, device)
        test_loss, hc_pd_test_pred, hc_pd_test_labels, hc_pd_test_probs, \
        pd_dd_test_pred, pd_dd_test_labels, pd_dd_test_probs = test_results
        
        final_metrics_hc = calculate_metrics(
            hc_pd_test_labels, hc_pd_test_pred,
            f"Outer Fold {outer_fold_idx + 1} - Test HC vs PD", verbose=True
        )
        final_metrics_pd = calculate_metrics(
            pd_dd_test_labels, pd_dd_test_pred,
            f"Outer Fold {outer_fold_idx + 1} - Test PD vs DD", verbose=True
        )
        
        test_features, test_hc_pd_labels, test_pd_dd_labels = extract_features(
            model, test_loader, device
        )
        
        outer_fold_result = {
            'outer_fold': outer_fold_idx + 1,
            'best_hyperparams': best_params,
            'best_test_acc': best_test_acc,
            'final_metrics_hc': final_metrics_hc,
            'final_metrics_pd': final_metrics_pd,
            'history': history
        }
        all_outer_results.append(outer_fold_result)
        
        # Save best model
        torch.save({
            'model_state_dict': best_model_state,
            'outer_fold': outer_fold_idx + 1,
            'best_test_acc': best_test_acc,
            'hyperparameters': best_params,
            'config': config
        }, f'checkpoints/best_model_nested_cv_fold_{outer_fold_idx + 1}.pth')
        print(f"✓ Model saved: checkpoints/best_model_nested_cv_fold_{outer_fold_idx + 1}.pth")
        
        # Save training history
        import pandas as pd
        history_df = pd.DataFrame(history)
        history_df.to_csv(f"training_history/fold_{outer_fold_idx + 1}_history.csv", index=False)
        
        # Mark fold as complete and save global run state
        mark_fold_done(outer_fold_idx)
        save_run_state(outer_fold_idx, all_outer_results, best_hyperparams_per_fold)
        print(f"  [ckpt] Outer fold {outer_fold_idx+1} marked as done.")
    
        # Generate plots
        if config.get('create_plots', True):
            plot_dir = f"plots/nested_cv_fold_{outer_fold_idx + 1}"
            os.makedirs(plot_dir, exist_ok=True)
            
            plot_loss(history['train_loss'], history['test_loss'], f"{plot_dir}/loss.png")
            
            if best_hc_pd_probs is not None and len(best_hc_pd_labels) > 0:
                plot_roc_curves(best_hc_pd_labels, best_hc_pd_preds, best_hc_pd_probs,
                              f"{plot_dir}/roc_hc_vs_pd.png")
            
            if best_pd_dd_probs is not None and len(best_pd_dd_labels) > 0:
                plot_roc_curves(best_pd_dd_labels, best_pd_dd_preds, best_pd_dd_probs,
                              f"{plot_dir}/roc_pd_vs_dd.png")
            
            plot_tsne(test_features, test_hc_pd_labels, test_pd_dd_labels, output_dir=plot_dir)
            
            print(f"✓ All plots saved to {plot_dir}")
    
    # Compute and save overall statistics
    all_test_accs = [r['best_test_acc'] for r in all_outer_results]
    all_hc_accs = [r['final_metrics_hc'].get('accuracy', 0) for r in all_outer_results]
    all_pd_accs = [r['final_metrics_pd'].get('accuracy', 0) for r in all_outer_results]
    
    print(f"\n" + "="*80)
    print(f"OVERALL STATISTICS:")
    print(f"  Combined Test Acc: {np.mean(all_test_accs):.4f} ± {np.std(all_test_accs):.4f}")
    print(f"  HC vs PD Acc: {np.mean(all_hc_accs):.4f} ± {np.std(all_hc_accs):.4f}")
    print(f"  PD vs DD Acc: {np.mean(all_pd_accs):.4f} ± {np.std(all_pd_accs):.4f}")
    
    # Save overall results to JSON
    overall_results = {
        'combined_test_acc_mean': float(np.mean(all_test_accs)),
        'combined_test_acc_std': float(np.std(all_test_accs)),
        'hc_vs_pd_acc_mean': float(np.mean(all_hc_accs)),
        'hc_vs_pd_acc_std': float(np.std(all_hc_accs)),
        'pd_vs_dd_acc_mean': float(np.mean(all_pd_accs)),
        'pd_vs_dd_acc_std': float(np.std(all_pd_accs)),
        'per_fold': [{
            'fold': r['outer_fold'],
            'best_test_acc': r['best_test_acc'],
            'hc_acc': r['final_metrics_hc'].get('accuracy', 0),
            'pd_acc': r['final_metrics_pd'].get('accuracy', 0),
            'hyperparams': r['best_hyperparams']
        } for r in all_outer_results]
    }
    with open('results/nested_cv_overall_results.json', 'w') as f:
        json.dump(overall_results, f, indent=2)
    print("✓ Overall results saved: results/nested_cv_overall_results.json")
    
    print("\n✓ All results, metrics, and plots saved successfully!")
    
    return all_outer_results

# ============================================================================
# MAIN FUNCTION 
# ============================================================================
def main():
    
    config = {
        'data_root': "/kaggle/input/datasets/meherujannat/parkinsons/pads-parkinsons-disease-smartwatch-dataset-1.0.0",
        'apply_downsampling': True,  
        'apply_bandpass_filter': True,
        
        # Nested CV settings
        'outer_folds': 5,         
        'inner_folds': 3,         
        'n_trials': 20,           
        'optuna_epochs': 20,      
        'final_epochs': 80,       

        # Model architecture
        'input_dim': 6,
        'timestep': 256,
        'num_classes': 2,
    
        # General
        'num_workers': 4,            # Kaggle typically has 4 CPU cores
        'pin_memory': True,          # Faster CPU→GPU transfers (auto-disabled on CPU)
        'save_metrics': True,
        'create_plots': True,

        # Resumable training
        'patience': 10,              # Early stopping patience for final training
        'checkpoint_interval': 5,    # Save mid-training checkpoint every N epochs
        'resume_from': "/kaggle/input/datasets/meheruzannat/fold-3/optuna3",  
    }
    
    results = train_model(config)
    
    return results


if __name__ == "__main__":
    results = main()