"""
Self-Supervised Contrastive Learning for Parkinson's Disease Detection
=======================================================================
This script implements self-supervised pre-training using contrastive loss,
followed by fine-tuning with full evaluation metrics (same as base model).
Includes: Stratified K-Fold, ROC curves, t-SNE plots, fold metrics saving.
"""

import pathlib
import numpy as np
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
import warnings
from scipy.signal import butter, filtfilt
from scipy.interpolate import CubicSpline
import os
import math
import csv
from collections import defaultdict
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.manifold import TSNE
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# ============================================================================
# AUGMENTATION FUNCTIONS
# ============================================================================

def add_gaussian_noise(data: np.ndarray, noise_level: float = 0.1) -> np.ndarray:
    """Add Gaussian noise to the signal."""
    noise = np.random.normal(0, noise_level * np.std(data), data.shape)
    return data + noise


def add_uniform_noise(data: np.ndarray, noise_level: float = 0.1) -> np.ndarray:
    """Add uniform noise to the signal."""
    data_range = np.max(data) - np.min(data)
    noise = np.random.uniform(-noise_level * data_range, noise_level * data_range, data.shape)
    return data + noise


def time_warp(data: np.ndarray, sigma: float = 0.2, num_knots: int = 4) -> np.ndarray:
    """Apply time warping augmentation using cubic spline interpolation."""
    timesteps, channels = data.shape
    orig_steps = np.arange(timesteps)
    
    knot_positions = np.linspace(0, timesteps - 1, num_knots + 2)
    random_warps = np.random.normal(loc=1.0, scale=sigma, size=num_knots + 2)
    random_warps[0] = 1.0
    random_warps[-1] = 1.0
    
    cumulative_warp = np.cumsum(random_warps)
    cumulative_warp = (cumulative_warp - cumulative_warp[0]) / (cumulative_warp[-1] - cumulative_warp[0])
    cumulative_warp = cumulative_warp * (timesteps - 1)
    
    warp_spline = CubicSpline(knot_positions, cumulative_warp)
    warped_steps = warp_spline(orig_steps)
    warped_steps = np.clip(warped_steps, 0, timesteps - 1)
    
    warped_data = np.zeros_like(data)
    for c in range(channels):
        channel_spline = CubicSpline(orig_steps, data[:, c])
        warped_data[:, c] = channel_spline(warped_steps)
    
    return warped_data


def magnitude_warp(data: np.ndarray, sigma: float = 0.2, num_knots: int = 4) -> np.ndarray:
    """Apply magnitude warping augmentation."""
    timesteps, channels = data.shape
    orig_steps = np.arange(timesteps)
    knot_positions = np.linspace(0, timesteps - 1, num_knots + 2)
    random_scales = np.random.normal(loc=1.0, scale=sigma, size=(num_knots + 2, channels))
    
    warped_data = np.zeros_like(data)
    for c in range(channels):
        scale_spline = CubicSpline(knot_positions, random_scales[:, c])
        scales = scale_spline(orig_steps)
        warped_data[:, c] = data[:, c] * scales
    
    return warped_data


def scaling(data: np.ndarray, sigma: float = 0.1) -> np.ndarray:
    """Apply random scaling to each channel."""
    scales = np.random.normal(loc=1.0, scale=sigma, size=(1, data.shape[1]))
    return data * scales


def permutation(data: np.ndarray, num_segments: int = 4) -> np.ndarray:
    """Randomly permute segments of the signal."""
    timesteps, channels = data.shape
    segment_length = timesteps // num_segments
    
    segments = []
    for i in range(num_segments):
        start = i * segment_length
        end = start + segment_length if i < num_segments - 1 else timesteps
        segments.append(data[start:end, :])
    
    np.random.shuffle(segments)
    return np.concatenate(segments, axis=0)


# ============================================================================
# HELPER FUNCTIONS (from base model)
# ============================================================================

def create_windows(data, window_size=256, overlap=0):
    n_samples, n_channels = data.shape
    step = int(window_size * (1 - overlap))   
    windows = []
    for start in range(0, n_samples - window_size + 1, step):
        end = start + window_size
        windows.append(data[start:end, :])
    return np.array(windows) if windows else None


def downsample(data, original_freq=100, target_freq=64):
    step = int(original_freq // target_freq)  
    if step > 1:
        return data[::step, :]
    return data


def bandpass_filter(signal, original_freq=64, upper_bound=20, lower_bound=0.1):
    nyquist = 0.5 * original_freq
    low = lower_bound / nyquist
    high = upper_bound / nyquist
    b, a = butter(5, [low, high], btype='band')
    return filtfilt(b, a, signal, axis=0)


# ============================================================================
# K-FOLD SPLIT METHOD (from base model)
# ============================================================================

def k_fold_split_method(data_root, full_dataset, k=5):
    """Stratified K-Fold split at patient level."""
    patient_conditions = {}
    patients_template = pathlib.Path(data_root) / "patients" / "patient_{p:03d}.json"
    
    for patient_id in range(1, 470):
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
# CONTRASTIVE DATASET (for pre-training)
# ============================================================================

class ContrastiveDataset(Dataset):
    """Dataset for self-supervised contrastive learning."""
    
    def __init__(
        self,
        data_root: str = None,
        window_size: int = 256,
        left_samples=None,
        right_samples=None,
        apply_dowsampling: bool = True,
        apply_bandpass_filter: bool = True,
        augmentation_strength: float = 0.5,
        augmentation_type: int = 1,  # 1=gaussian_noise, 2=time_warp, 3=both_random
        **kwargs
    ):
        self.left_samples = []
        self.right_samples = []
        self.patient_ids = []
        self.task_names = []
        self.apply_dowsampling = apply_dowsampling
        self.apply_bandpass_filter = apply_bandpass_filter
        self.augmentation_strength = augmentation_strength
        self.augmentation_type = augmentation_type
        self.data_root = data_root
        self.window_size = window_size

        if data_root is not None:
            self.patients_template = pathlib.Path(data_root) / "patients" / "patient_{p:03d}.json"
            self.timeseries_template = pathlib.Path(data_root) / "movement" / "timeseries" / "{N:03d}_{X}_{Y}.txt"
            
            self.tasks = ["CrossArms", "DrinkGlas", "Entrainment", "HoldWeight", "LiftHold", 
                         "PointFinger", "Relaxed", "StretchHold", "TouchIndex", "TouchNose"]
            
            self.patient_ids_list = list(range(1, 470))
            print(f"Dataset: {len(self.patient_ids_list)} patients (001-469)")
            self._load_data()
        else:
            if left_samples is not None:
                self.left_samples = np.array(left_samples) if not isinstance(left_samples, np.ndarray) else left_samples
            if right_samples is not None:
                self.right_samples = np.array(right_samples) if not isinstance(right_samples, np.ndarray) else right_samples
            self.patient_ids = kwargs.get('patient_ids', [])
            if self.patient_ids is not None and len(self.patient_ids) > 0:
                self.patient_ids = np.array(self.patient_ids) if not isinstance(self.patient_ids, np.ndarray) else self.patient_ids

    def _load_data(self):
        """Load all data without labels (self-supervised doesn't need labels)."""
        for patient_id in tqdm(self.patient_ids_list, desc="Loading patients"):
            patient_path = pathlib.Path(str(self.patients_template).format(p=patient_id))
            
            if not patient_path.exists():
                continue
                
            try:
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
                            left_data = left_data[:, :6]
                        if left_data.shape[0] > 50:
                            left_data = left_data[50:, :]
                        
                        if right_data.shape[1] > 6:
                            right_data = right_data[:, :6]
                        if right_data.shape[0] > 50:
                            right_data = right_data[50:, :]
                        
                        if self.apply_dowsampling:
                            left_data = downsample(left_data)
                            right_data = downsample(right_data)
                            
                        if self.apply_bandpass_filter:
                            left_data = bandpass_filter(left_data)
                            right_data = bandpass_filter(right_data)

                        if left_data is None or right_data is None:
                            continue
                        
                        overlap = 0.5
                        left_windows = create_windows(left_data, self.window_size, overlap=overlap)
                        right_windows = create_windows(right_data, self.window_size, overlap=overlap)

                        if left_windows is not None and right_windows is not None:
                            min_windows = min(len(left_windows), len(right_windows))
                            
                            for i in range(min_windows):
                                patient_left_samples.append(left_windows[i])
                                patient_right_samples.append(right_windows[i])
                                patient_task_names.append(task)
                        
                    except Exception as e:
                        continue
                
                if len(patient_left_samples) > 0:
                    for i in range(len(patient_left_samples)):
                        self.left_samples.append(patient_left_samples[i])
                        self.right_samples.append(patient_right_samples[i])
                        self.patient_ids.append(patient_id)
                        self.task_names.append(patient_task_names[i])
                
            except Exception as e:
                continue
        
        self.left_samples = np.array(self.left_samples)
        self.right_samples = np.array(self.right_samples)
        self.patient_ids = np.array(self.patient_ids)
        self.task_names = np.array(self.task_names)
        
        print(f"Loaded {len(self.left_samples)} samples for self-supervised learning")

    def __len__(self):
        return len(self.left_samples)
    
    def __getitem__(self, idx):
        """Returns anchor, positive (pure/same as anchor), and negative (augmented) pairs."""
        left_anchor = self.left_samples[idx].copy()
        right_anchor = self.right_samples[idx].copy()
        
        # Positive pair (pure - no augmentation, same as anchor)
        left_positive = left_anchor.copy()
        right_positive = right_anchor.copy()
        
        # Negative pair (augmented based on augmentation_type flag)
        # 1=gaussian_noise, 2=time_warp, 3=both_random
        if self.augmentation_type == 1:
            left_negative = add_gaussian_noise(left_anchor, noise_level=0.3)
            right_negative = add_gaussian_noise(right_anchor, noise_level=0.3)
        elif self.augmentation_type == 2:
            left_negative = time_warp(left_anchor, sigma=0.3)
            right_negative = time_warp(right_anchor, sigma=0.3)
        else:  # augmentation_type == 3: randomly choose between noise and time_warp
            if np.random.random() > 0.5:
                left_negative = add_gaussian_noise(left_anchor, noise_level=0.3)
                right_negative = add_gaussian_noise(right_anchor, noise_level=0.3)
            else:
                left_negative = time_warp(left_anchor, sigma=0.3)
                right_negative = time_warp(right_anchor, sigma=0.3)
        
        return (
            torch.FloatTensor(left_anchor),
            torch.FloatTensor(right_anchor),
            torch.FloatTensor(left_positive),
            torch.FloatTensor(right_positive),
            torch.FloatTensor(left_negative),
            torch.FloatTensor(right_negative)
        )


# ============================================================================
# LABELED DATASET (for fine-tuning - same as base model)
# ============================================================================

class ParkinsonsDataLoader(Dataset):
    """Dataset with labels for fine-tuning."""
    
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
            
            self.tasks = ["CrossArms", "DrinkGlas", "Entrainment", "HoldWeight", "LiftHold", 
                         "PointFinger", "Relaxed", "StretchHold", "TouchIndex", "TouchNose"]
            
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
            
            if not patient_path.exists():
                continue
                
            try:
                with open(patient_path, 'r') as f:
                    metadata = json.load(f)
                
                condition = metadata.get('condition', '')
            
                if condition == 'Healthy':
                    hc_vs_pd_label = 0
                    pd_vs_dd_label = -1
                    overlap = 0.70
                elif 'Parkinson' in condition:
                    hc_vs_pd_label = 1
                    pd_vs_dd_label = 0
                    overlap = 0
                else:
                    hc_vs_pd_label = -1
                    pd_vs_dd_label = 1
                    overlap = 0.65

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
                            left_data = left_data[:, :6]
                        if left_data.shape[0] > 50:
                            left_data = left_data[50:, :]
                        
                        if right_data.shape[1] > 6:
                            right_data = right_data[:, :6]
                        if right_data.shape[0] > 50:
                            right_data = right_data[50:, :]
                        
                        if self.apply_dowsampling:
                            left_data = downsample(left_data)
                            right_data = downsample(right_data)
                            
                        if self.apply_bandpass_filter:
                            left_data = bandpass_filter(left_data)
                            right_data = bandpass_filter(right_data)

                        if left_data is None or right_data is None:
                            continue
                        
                        left_windows = create_windows(left_data, self.window_size, overlap=overlap)
                        right_windows = create_windows(right_data, self.window_size, overlap=overlap)

                        if left_windows is not None and right_windows is not None:
                            min_windows = min(len(left_windows), len(right_windows))
                            
                            for i in range(min_windows):
                                patient_left_samples.append(left_windows[i])
                                patient_right_samples.append(right_windows[i])
                                patient_task_names.append(task)
                        
                    except Exception as e:
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
                continue
        
        self.left_samples = np.array(self.left_samples)
        self.right_samples = np.array(self.right_samples)
        self.hc_vs_pd = np.array(self.hc_vs_pd)
        self.pd_vs_dd = np.array(self.pd_vs_dd)
        self.patient_ids = np.array(self.patient_ids)
        self.task_names = np.array(self.task_names)

    def get_train_test_split(self, split_type=3, **kwargs):
        if split_type == 3:
            k = kwargs.get('k', 5)
            if self.data_root is None:
                raise ValueError("data_root is required for K-fold split")
            fold_datasets = k_fold_split_method(self.data_root, self, k)
            return fold_datasets
        else:
            raise ValueError(f"Invalid split_type: {split_type}")

    def __len__(self):
        return len(self.left_samples) if hasattr(self, 'left_samples') and isinstance(self.left_samples, (list, np.ndarray)) else 0
    
    def __getitem__(self, idx):
        left_sample = torch.FloatTensor(self.left_samples[idx])
        right_sample = torch.FloatTensor(self.right_samples[idx])
        hc_vs_pd = torch.LongTensor([self.hc_vs_pd[idx]])
        pd_vs_dd = torch.LongTensor([self.pd_vs_dd[idx]])
        
        return left_sample, right_sample, hc_vs_pd.squeeze(), pd_vs_dd.squeeze()


# ============================================================================
# MODEL COMPONENTS
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
        
        self.cross_attention_1to2 = nn.MultiheadAttention(embed_dim=model_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.cross_attention_2to1 = nn.MultiheadAttention(embed_dim=model_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        
        self.self_attention_1 = nn.MultiheadAttention(embed_dim=model_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.self_attention_2 = nn.MultiheadAttention(embed_dim=model_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        
        self.norm_cross_1 = nn.LayerNorm(model_dim)
        self.norm_cross_2 = nn.LayerNorm(model_dim)
        self.norm_self_1 = nn.LayerNorm(model_dim)
        self.norm_self_2 = nn.LayerNorm(model_dim)
        
        self.feed_forward_1 = FeedForward(model_dim, d_ff, dropout)       
        self.feed_forward_2 = FeedForward(model_dim, d_ff, dropout)
        
    def forward(self, channel_1, channel_2):
        channel_1_cross_attn, _ = self.cross_attention_1to2(query=channel_1, key=channel_2, value=channel_2)
        channel_1_cross = self.norm_cross_1(channel_1 + channel_1_cross_attn)
        
        channel_2_cross_attn, _ = self.cross_attention_2to1(query=channel_2, key=channel_1, value=channel_1)
        channel_2_cross = self.norm_cross_2(channel_2 + channel_2_cross_attn)
        
        channel_1_self_attn, _ = self.self_attention_1(query=channel_1_cross, key=channel_1_cross, value=channel_1_cross)
        channel_1_self = self.norm_self_1(channel_1_cross + channel_1_self_attn)
        
        channel_2_self_attn, _ = self.self_attention_2(query=channel_2_cross, key=channel_2_cross, value=channel_2_cross)
        channel_2_self = self.norm_self_2(channel_2_cross + channel_2_self_attn)
        
        channel_1_out = self.feed_forward_1(channel_1_self)
        channel_2_out = self.feed_forward_2(channel_2_self)

        return channel_1_out, channel_2_out


class ContrastiveEncoder(nn.Module):
    """Encoder for self-supervised contrastive learning."""
    
    def __init__(
        self,
        input_dim: int = 6,
        model_dim: int = 128,
        num_heads: int = 8,
        num_layers: int = 4,
        d_ff: int = 512,
        dropout: float = 0.1,
        timestep: int = 256,
        projection_dim: int = 128,
    ):
        super().__init__()
        
        self.model_dim = model_dim
        self.timestep = timestep
        
        self.left_projection = nn.Linear(input_dim, model_dim)
        self.right_projection = nn.Linear(input_dim, model_dim)
        
        self.positional_encoding = PositionalEncoding(model_dim, max_len=timestep)
        
        self.layers = nn.ModuleList([
            CrossAttention(model_dim, num_heads, d_ff, dropout) 
            for _ in range(num_layers)
        ])
            
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        
        feature_dim = model_dim * 2
        
        self.projection_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, projection_dim)
        )
        
    def encode(self, left_wrist, right_wrist):
        """Get features before projection head."""
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

        features = torch.cat([left_pool, right_pool], dim=1)
        
        return features
    
    def forward(self, left_wrist, right_wrist):
        """Forward pass with projection for contrastive loss."""
        features = self.encode(left_wrist, right_wrist)
        projections = self.projection_head(features)
        embeddings = F.normalize(projections, p=2, dim=1)
        return embeddings
    
    def get_features(self, left_wrist, right_wrist):
        return self.encode(left_wrist, right_wrist)


class MainModel(nn.Module):
    """Classification model for fine-tuning (same as base model)."""
    
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
    ):
        super().__init__()
        
        self.model_dim = model_dim
        self.timestep = timestep
        
        self.left_projection = nn.Linear(input_dim, model_dim)
        self.right_projection = nn.Linear(input_dim, model_dim)
        
        self.positional_encoding = PositionalEncoding(model_dim, max_len=timestep)
        
        self.layers = nn.ModuleList([
            CrossAttention(model_dim, num_heads, d_ff, dropout) 
            for _ in range(num_layers)
        ])
            
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        fusion_dim = model_dim * 2

        self.head_hc_vs_pd = nn.Sequential(
            nn.Linear(fusion_dim, model_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim, 2)
        )
        
        self.head_pd_vs_dd = nn.Sequential(
            nn.Linear(fusion_dim, model_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim, 2)
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

        fused_features = torch.cat([left_pool, right_pool], dim=1)
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

        fused_features = torch.cat([left_pool, right_pool], dim=1)

        logits_hc_vs_pd = self.head_hc_vs_pd(fused_features)
        logits_pd_vs_dd = self.head_pd_vs_dd(fused_features)

        return logits_hc_vs_pd, logits_pd_vs_dd


# ============================================================================
# CONTRASTIVE LOSS FUNCTIONS
# ============================================================================

class ContrastiveLoss(nn.Module):
    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin
    
    def forward(self, anchor, positive, negative):
        pos_distance = F.pairwise_distance(anchor, positive, p=2)
        neg_distance = F.pairwise_distance(anchor, negative, p=2)
        loss = torch.mean(torch.clamp(self.margin + pos_distance - neg_distance, min=0))
        return loss


class TripletLoss(nn.Module):
    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.triplet_loss = nn.TripletMarginLoss(margin=margin, p=2)
    
    def forward(self, anchor, positive, negative):
        return self.triplet_loss(anchor, positive, negative)


class NTXentLoss(nn.Module):
    def __init__(self, temperature: float = 0.5):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, anchor, positive, negative):
        batch_size = anchor.size(0)
        pos_sim = F.cosine_similarity(anchor, positive, dim=1) / self.temperature
        neg_sim = F.cosine_similarity(anchor, negative, dim=1) / self.temperature
        anchor_sim = torch.mm(anchor, anchor.t()) / self.temperature
        mask = torch.eye(batch_size, device=anchor.device).bool()
        anchor_sim = anchor_sim.masked_fill(mask, float('-inf'))
        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim.unsqueeze(1), anchor_sim], dim=1)
        labels = torch.zeros(batch_size, dtype=torch.long, device=anchor.device)
        loss = F.cross_entropy(logits, labels)
        return loss


class InfoNCELoss(nn.Module):
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, anchor, positive, negative):
        batch_size = anchor.size(0)
        pos_sim = torch.sum(anchor * positive, dim=1, keepdim=True) / self.temperature
        neg_sim_explicit = torch.sum(anchor * negative, dim=1, keepdim=True) / self.temperature
        neg_sim_batch = torch.mm(anchor, anchor.t()) / self.temperature
        mask = torch.eye(batch_size, device=anchor.device).bool()
        neg_sim_batch = neg_sim_batch.masked_fill(mask, float('-inf'))
        logits = torch.cat([pos_sim, neg_sim_explicit, neg_sim_batch], dim=1)
        labels = torch.zeros(batch_size, dtype=torch.long, device=anchor.device)
        loss = F.cross_entropy(logits, labels)
        return loss


# ============================================================================
# EVALUATION FUNCTIONS (from base model)
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
        print(f"Precision: {precision_avg:.4f}")
        print(f"Recall: {recall_avg:.4f}")
        print(f"F1: {f1_avg:.4f}")
        
        unique_labels = np.unique(np.concatenate([y_true, y_pred]))
        for i, label in enumerate(unique_labels):
            if i < len(precision):
                label_name = "HC" if label == 0 else ("PD" if label == 1 else f"Class_{label}")
                if "PD vs DD" in task_name:
                    label_name = "PD" if label == 0 else ("DD" if label == 1 else f"Class_{label}")
                print(f"  {label_name}: Precision={precision[i]:.4f}, Recall={recall[i]:.4f}, F1={f1[i]:.4f}, Support={support[i]}")
        
        print("Confusion Matrix:")
        print(cm)
    
    return metrics


def save_fold_metric(fold_idx, fold_suffix, best_epoch, best_val_acc, fold_metrics_hc, fold_metrics_pd):
    os.makedirs("metrics", exist_ok=True)

    def write_csv(filename, metrics_list):
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["epoch", "accuracy", "precision", "recall", "f1"])
            for epoch_data in metrics_list:
                writer.writerow([
                    epoch_data['epoch'],
                    epoch_data['metrics'].get('accuracy', 0),
                    epoch_data['metrics'].get('precision_avg', 0),
                    epoch_data['metrics'].get('recall_avg', 0),
                    epoch_data['metrics'].get('f1_avg', 0)
                ])

    if fold_metrics_hc:
        hc_filename = f"metrics/ssl_hc_vs_pd_metrics{fold_suffix}.csv"
        write_csv(hc_filename, fold_metrics_hc)
        print(f"✓ HC vs PD metrics saved: {hc_filename}")

    if fold_metrics_pd:
        pd_filename = f"metrics/ssl_pd_vs_dd_metrics{fold_suffix}.csv"
        write_csv(pd_filename, fold_metrics_pd)
        print(f"✓ PD vs DD metrics saved: {pd_filename}")


def plot_loss(train_losses, val_losses, output_path):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss', color='blue')
    plt.plot(val_losses, label='Validation Loss', color='orange')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training and Validation Loss over Epochs', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Loss plot saved: {output_path}")


def plot_roc_curves(labels, predictions, probabilities, output_path):
    plt.figure(figsize=(10, 8))
    
    fpr, tpr, _ = roc_curve(labels, probabilities)
    roc_auc = auc(fpr, tpr)
    
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
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
    print(f"✓ ROC curve saved: {output_path}")


def plot_tsne(features, hc_pd_labels, pd_dd_labels, output_dir="plots"):
    if features is None or len(features) == 0:
        print("No features available for t-SNE visualization")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("Performing t-SNE dimensionality reduction...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    features_2d = tsne.fit_transform(features)

    valid_hc_pd = hc_pd_labels != -1
    valid_pd_dd = pd_dd_labels != -1

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

        plt.title("t-SNE: HC vs PD (Self-Supervised)")
        plt.xlabel("t-SNE Component 1")
        plt.ylabel("t-SNE Component 2")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, "tsne_hc_vs_pd.png"), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ t-SNE HC vs PD saved: {output_dir}/tsne_hc_vs_pd.png")

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

        plt.title("t-SNE: PD vs DD (Self-Supervised)")
        plt.xlabel("t-SNE Component 1")
        plt.ylabel("t-SNE Component 2")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, "tsne_pd_vs_dd.png"), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ t-SNE PD vs DD saved: {output_dir}/tsne_pd_vs_dd.png")

    return features_2d


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def train_contrastive_epoch(model, dataloader, criterion, optimizer, device):
    """Train encoder for one epoch with contrastive loss."""
    model.train()
    total_loss = 0.0
    
    for batch in tqdm(dataloader, desc="Contrastive Training"):
        (left_anchor, right_anchor, 
         left_positive, right_positive,
         left_negative, right_negative) = batch
        
        left_anchor = left_anchor.to(device)
        right_anchor = right_anchor.to(device)
        left_positive = left_positive.to(device)
        right_positive = right_positive.to(device)
        left_negative = left_negative.to(device)
        right_negative = right_negative.to(device)
        
        optimizer.zero_grad()
        
        anchor_emb = model(left_anchor, right_anchor)
        positive_emb = model(left_positive, right_positive)
        negative_emb = model(left_negative, right_negative)
        
        loss = criterion(anchor_emb, positive_emb, negative_emb)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def validate_contrastive_epoch(model, dataloader, criterion, device):
    """Validate contrastive learning."""
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Contrastive Validation"):
            (left_anchor, right_anchor, 
             left_positive, right_positive,
             left_negative, right_negative) = batch
            
            left_anchor = left_anchor.to(device)
            right_anchor = right_anchor.to(device)
            left_positive = left_positive.to(device)
            right_positive = right_positive.to(device)
            left_negative = left_negative.to(device)
            right_negative = right_negative.to(device)
            
            anchor_emb = model(left_anchor, right_anchor)
            positive_emb = model(left_positive, right_positive)
            negative_emb = model(left_negative, right_negative)
            
            loss = criterion(anchor_emb, positive_emb, negative_emb)
            total_loss += loss.item()
    
    return total_loss / len(dataloader)


def train_single_epoch(model, dataloader, criterion_hc, criterion_pd, optimizer, device):
    """Train classifier for one epoch (same as base model)."""
    model.train()
    train_loss = 0.0
    hc_pd_train_pred, hc_pd_train_labels = [], []
    pd_dd_train_pred, pd_dd_train_labels = [], []
    
    for batch in tqdm(dataloader, desc="Training"):
        left_sample, right_sample, hc_pd, pd_dd = batch
        
        left_sample = left_sample.to(device)
        right_sample = right_sample.to(device)
        hc_pd = hc_pd.to(device)
        pd_dd = pd_dd.to(device)
        
        optimizer.zero_grad()
        hc_pd_logits, pd_dd_logits = model(left_sample, right_sample, device) 
        
        total_loss = 0
        loss_count = 0
        
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
        
        if loss_count > 0:
            avg_loss = total_loss / loss_count
            avg_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += avg_loss.item()
    
    train_loss /= len(dataloader)
    
    train_metrics_hc = calculate_metrics(hc_pd_train_labels, hc_pd_train_pred, "Training HC vs PD", verbose=False)
    train_metrics_pd = calculate_metrics(pd_dd_train_labels, pd_dd_train_pred, "Training PD vs DD", verbose=False)
    
    return train_loss, train_metrics_hc, train_metrics_pd


def validate_single_epoch(model, dataloader, criterion_hc, criterion_pd, device):
    """Validate classifier for one epoch (same as base model)."""
    model.eval()
    val_loss = 0.0
    hc_pd_val_pred, hc_pd_val_labels, hc_pd_val_probs = [], [], []
    pd_dd_val_pred, pd_dd_val_labels, pd_dd_val_probs = [], [], []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            left_sample, right_sample, hc_pd, pd_dd = batch
            
            left_sample = left_sample.to(device)
            right_sample = right_sample.to(device)
            hc_pd = hc_pd.to(device)
            pd_dd = pd_dd.to(device)
            
            hc_pd_logits, pd_dd_logits = model(left_sample, right_sample, device)  
            
            total_loss = 0
            loss_count = 0
            
            valid_hc_pd_mask = (hc_pd != -1)
            if valid_hc_pd_mask.any():
                valid_logits_hc = hc_pd_logits[valid_hc_pd_mask]
                valid_labels_hc = hc_pd[valid_hc_pd_mask]
                loss_hc = criterion_hc(valid_logits_hc, valid_labels_hc)
                total_loss += loss_hc
                loss_count += 1
                
                preds_hc = torch.argmax(valid_logits_hc, dim=1)
                probs_hc = F.softmax(valid_logits_hc, dim=1)[:, 1]
                hc_pd_val_pred.extend(preds_hc.cpu().numpy())
                hc_pd_val_labels.extend(valid_labels_hc.cpu().numpy())
                hc_pd_val_probs.extend(probs_hc.cpu().numpy())
            
            valid_pd_dd_mask = (pd_dd != -1)
            if valid_pd_dd_mask.any():
                valid_logits_pd = pd_dd_logits[valid_pd_dd_mask]
                valid_labels_pd = pd_dd[valid_pd_dd_mask]
                loss_pd = criterion_pd(valid_logits_pd, valid_labels_pd)
                total_loss += loss_pd
                loss_count += 1
                
                preds_pd = torch.argmax(valid_logits_pd, dim=1)
                probs_pd = F.softmax(valid_logits_pd, dim=1)[:, 1]
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
    """Extract features for t-SNE visualization."""
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


def transfer_weights_to_classifier(pretrained_encoder, classifier_model):
    """Transfer weights from pre-trained encoder to classifier model."""
    classifier_model.left_projection.load_state_dict(pretrained_encoder.left_projection.state_dict())
    classifier_model.right_projection.load_state_dict(pretrained_encoder.right_projection.state_dict())
    classifier_model.positional_encoding.load_state_dict(pretrained_encoder.positional_encoding.state_dict())
    
    for pretrained_layer, classifier_layer in zip(pretrained_encoder.layers, classifier_model.layers):
        classifier_layer.load_state_dict(pretrained_layer.state_dict())
    
    print("✓ Transferred pre-trained weights to classifier model")
    return classifier_model


# ============================================================================
# MAIN TRAINING PIPELINE
# ============================================================================

def pretrain_self_supervised(config):
    """Phase 1: Self-supervised pre-training with contrastive loss."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("plots/pretrain", exist_ok=True)
    
    print("\n" + "="*60)
    print("PHASE 1: Self-Supervised Pre-training")
    print("="*60)
    
    full_dataset = ContrastiveDataset(
        data_root=config['data_root'],
        window_size=config.get('window_size', 256),
        apply_dowsampling=config.get('apply_downsampling', True),
        apply_bandpass_filter=config.get('apply_bandpass_filter', True),
        augmentation_strength=config.get('augmentation_strength', 0.5),
        augmentation_type=config.get('augmentation_type', 1)
    )
    
    train_size = int(0.85 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"Pre-training samples - Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=config['pretrain_batch_size'], shuffle=True, num_workers=config.get('num_workers', 0))
    val_loader = DataLoader(val_dataset, batch_size=config['pretrain_batch_size'], shuffle=False, num_workers=config.get('num_workers', 0))
    
    model = ContrastiveEncoder(
        input_dim=config.get('input_dim', 6),
        model_dim=config.get('model_dim', 32),
        num_heads=config.get('num_heads', 8),
        num_layers=config.get('num_layers', 3),
        d_ff=config.get('d_ff', 256),
        dropout=config.get('dropout', 0.1),
        timestep=config.get('timestep', 256),
        projection_dim=config.get('projection_dim', 64)
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    optimizer = optim.AdamW(model.parameters(), lr=config.get('pretrain_lr', 1e-4), weight_decay=config.get('weight_decay', 1e-4))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['pretrain_epochs'], eta_min=1e-6)
    
    loss_type = config.get('loss_type', 'triplet')
    if loss_type == 'contrastive':
        criterion = ContrastiveLoss(margin=config.get('margin', 1.0))
    elif loss_type == 'triplet':
        criterion = TripletLoss(margin=config.get('margin', 1.0))
    elif loss_type == 'ntxent':
        criterion = NTXentLoss(temperature=config.get('temperature', 0.5))
    elif loss_type == 'infonce':
        criterion = InfoNCELoss(temperature=config.get('temperature', 0.07))
    
    print(f"Using {loss_type} loss")
    
    history = {'train_loss': [], 'val_loss': []}
    best_val_loss = float('inf')
    patience_counter = 0
    patience = config.get('early_stopping_patience', 15)
    
    for epoch in range(config['pretrain_epochs']):
        print(f"\nPre-train Epoch {epoch + 1}/{config['pretrain_epochs']}")
        
        train_loss = train_contrastive_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate_contrastive_epoch(model, val_loader, criterion, device)
        
        scheduler.step()
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'config': config
            }, 'checkpoints/best_contrastive_model.pth')
            print(f"✓ Saved best pre-trained model (val_loss: {val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch + 1}")
                break
    
    # Plot pre-training loss
    plot_loss(history['train_loss'], history['val_loss'], 'plots/pretrain/contrastive_loss.png')
    
    # Load best model
    checkpoint = torch.load('checkpoints/best_contrastive_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"\n✓ Pre-training complete! Best val_loss: {best_val_loss:.4f}")
    
    return model, history


def finetune_with_kfold(pretrained_encoder, config):
    """Phase 2: Fine-tune with stratified K-fold and full evaluation."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    os.makedirs("metrics", exist_ok=True)
    
    print("\n" + "="*60)
    print("PHASE 2: Fine-tuning with Stratified K-Fold")
    print("="*60)
    
    # Load labeled dataset
    full_dataset = ParkinsonsDataLoader(
        config['data_root'],
        apply_dowsampling=config['apply_downsampling'],
        apply_bandpass_filter=config['apply_bandpass_filter']
    )
    
    fold_datasets = full_dataset.get_train_test_split(split_type=3, k=config['num_folds'])
    num_folds = len(fold_datasets)
    
    all_fold_results = []
    
    for fold_idx in range(num_folds):
        print(f"\n{'='*60}")
        print(f"Starting Fold {fold_idx+1}/{num_folds}")
        print(f"{'='*60}")
        
        train_dataset, val_dataset = fold_datasets[fold_idx]
        
        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=config['num_workers'])
        val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers'])
        
        # Create classifier and transfer pre-trained weights
        model = MainModel(
            input_dim=config['input_dim'],
            model_dim=config['model_dim'],
            num_heads=config['num_heads'],
            num_layers=config['num_layers'],
            d_ff=config['d_ff'],
            dropout=config['dropout'],
            timestep=config['timestep'],
            num_classes=config['num_classes']
        ).to(device)
        
        model = transfer_weights_to_classifier(pretrained_encoder, model)
        
        optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
        
        hc_pd_loss = nn.CrossEntropyLoss()
        pd_dd_loss = nn.CrossEntropyLoss()
        
        history = defaultdict(list)
        best_val_acc = 0.0
        best_epoch = 0
        
        fold_metrics_hc = []
        fold_metrics_pd = []
        
        best_hc_pd_probs = None
        best_hc_pd_preds = None
        best_hc_pd_labels = None
        best_pd_dd_probs = None
        best_pd_dd_preds = None
        best_pd_dd_labels = None
        
        for epoch in range(config['num_epochs']):
            print(f"\nEpoch {epoch + 1}/{config['num_epochs']}")
            
            train_loss, train_metrics_hc, train_metrics_pd = train_single_epoch(
                model, train_loader, hc_pd_loss, pd_dd_loss, optimizer, device
            )
            
            val_results = validate_single_epoch(
                model, val_loader, hc_pd_loss, pd_dd_loss, device
            )
            val_loss, hc_pd_val_pred, hc_pd_val_labels, hc_pd_val_probs, \
            pd_dd_val_pred, pd_dd_val_labels, pd_dd_val_probs = val_results
            
            print("\n" + "="*60)
            val_metrics_hc = calculate_metrics(
                hc_pd_val_labels, hc_pd_val_pred,
                f"Fold {fold_idx+1} Validation HC vs PD", verbose=True
            )
            val_metrics_pd = calculate_metrics(
                pd_dd_val_labels, pd_dd_val_pred,
                f"Fold {fold_idx+1} Validation PD vs DD", verbose=True
            )
            print("="*60)
            
            if hc_pd_val_labels:
                fold_metrics_hc.append({
                    'epoch': epoch + 1,
                    'predictions': list(hc_pd_val_pred),
                    'labels': list(hc_pd_val_labels),
                    'metrics': val_metrics_hc
                })
            
            if pd_dd_val_labels:
                fold_metrics_pd.append({
                    'epoch': epoch + 1,
                    'predictions': list(pd_dd_val_pred),
                    'labels': list(pd_dd_val_labels),
                    'metrics': val_metrics_pd
                })
            
            val_acc_hc = val_metrics_hc.get('accuracy', 0)
            val_acc_pd = val_metrics_pd.get('accuracy', 0)
            val_acc_combined = (val_acc_hc + val_acc_pd) / 2
            
            train_acc_hc = train_metrics_hc.get('accuracy', 0)
            train_acc_pd = train_metrics_pd.get('accuracy', 0)
            
            scheduler.step(val_loss)
            
            history['train_loss'].append(train_loss)
            history['train_acc_hc'].append(train_acc_hc)
            history['train_acc_pd'].append(train_acc_pd)
            history['val_loss'].append(val_loss)
            history['val_acc_hc'].append(val_acc_hc)
            history['val_acc_pd'].append(val_acc_pd)
            history['val_acc_combined'].append(val_acc_combined)
            
            print(f"\nFold {fold_idx+1}, Epoch {epoch+1} Summary:")
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Train Acc - HC vs PD: {train_acc_hc:.4f}, PD vs DD: {train_acc_pd:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            print(f"Val Acc - HC vs PD: {val_acc_hc:.4f}, PD vs DD: {val_acc_pd:.4f}, Combined: {val_acc_combined:.4f}")
            
            if val_acc_combined > best_val_acc:
                best_val_acc = val_acc_combined
                best_epoch = epoch + 1
                
                if hc_pd_val_probs:
                    best_hc_pd_probs = np.array(hc_pd_val_probs)
                    best_hc_pd_preds = np.array(hc_pd_val_pred)
                    best_hc_pd_labels = np.array(hc_pd_val_labels)
                
                if pd_dd_val_probs:
                    best_pd_dd_probs = np.array(pd_dd_val_probs)
                    best_pd_dd_preds = np.array(pd_dd_val_pred)
                    best_pd_dd_labels = np.array(pd_dd_val_labels)
                
                model_save_name = f'checkpoints/ssl_best_model_fold_{fold_idx+1}.pth'
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'fold': fold_idx,
                    'epoch': epoch,
                    'val_acc_combined': val_acc_combined,
                    'val_acc_hc': val_acc_hc,
                    'val_acc_pd': val_acc_pd,
                    'config': config
                }, model_save_name)
                print(f"✓ New best model saved: {model_save_name}")
        
        # Save fold metrics
        if config.get('save_metrics', True):
            fold_suffix = f"_fold_{fold_idx+1}"
            if fold_metrics_hc and fold_metrics_pd:
                save_fold_metric(fold_idx, fold_suffix, best_epoch, best_val_acc, fold_metrics_hc, fold_metrics_pd)
        
        # Extract features and create plots
        fold_features, fold_hc_pd_labels, fold_pd_dd_labels = extract_features(model, val_loader, device)
        
        fold_result = {
            'fold': fold_idx + 1,
            'best_val_accuracy': best_val_acc,
            'best_epoch': best_epoch,
            'history': dict(history),
            'features': fold_features,
            'hc_pd_labels': fold_hc_pd_labels,
            'pd_dd_labels': fold_pd_dd_labels
        }
        all_fold_results.append(fold_result)
        
        # Create plots for this fold
        if config.get('create_plots', True):
            plot_dir = f"plots/ssl_fold_{fold_idx+1}"
            os.makedirs(plot_dir, exist_ok=True)
            
            plot_loss(history['train_loss'], history['val_loss'], f"{plot_dir}/loss.png")
            
            if best_hc_pd_probs is not None and len(best_hc_pd_labels) > 0:
                plot_roc_curves(best_hc_pd_labels, best_hc_pd_preds, best_hc_pd_probs, f"{plot_dir}/roc_hc_vs_pd.png")
            
            if best_pd_dd_probs is not None and len(best_pd_dd_labels) > 0:
                plot_roc_curves(best_pd_dd_labels, best_pd_dd_preds, best_pd_dd_probs, f"{plot_dir}/roc_pd_vs_dd.png")
            
            if fold_features is not None:
                plot_tsne(fold_features, fold_hc_pd_labels, fold_pd_dd_labels, output_dir=plot_dir)
    
    # Print summary
    print("\n" + "="*60)
    print("TRAINING COMPLETE - Summary")
    print("="*60)
    for result in all_fold_results:
        print(f"Fold {result['fold']}: Best Val Acc = {result['best_val_accuracy']:.4f} (Epoch {result['best_epoch']})")
    
    avg_acc = np.mean([r['best_val_accuracy'] for r in all_fold_results])
    std_acc = np.std([r['best_val_accuracy'] for r in all_fold_results])
    print(f"\nAverage Val Accuracy: {avg_acc:.4f} ± {std_acc:.4f}")
    
    return all_fold_results


def main():
    """Main function for self-supervised learning pipeline."""
    
    config = {
        # Data settings
        'data_root': "/kaggle/input/parkinsons/pads-parkinsons-disease-smartwatch-dataset-1.0.0",
        'apply_downsampling': True,
        'apply_bandpass_filter': True,
        'window_size': 256,
        'augmentation_strength': 0.5,
        'augmentation_type': 1,  # 1=gaussian_noise, 2=time_warp, 3=both_random
        
        # Model settings
        'input_dim': 6,
        'model_dim': 32,
        'num_heads': 8,
        'num_layers': 3,
        'd_ff': 256,
        'dropout': 0.12281570220908891,
        'timestep': 256,
        'num_classes': 2,
        'projection_dim': 64,
        
        # Pre-training settings
        'pretrain_batch_size': 64,
        'pretrain_lr': 1e-4,
        'pretrain_epochs': 50,
        'loss_type': 'triplet',  # 'contrastive', 'triplet', 'ntxent', 'infonce'
        'margin': 1.0,
        'temperature': 0.07,
        
        # Fine-tuning settings
        'num_folds': 5,
        'batch_size': 32,
        'learning_rate': 0.0002912623775216651,
        'weight_decay': 0.00016228510005606125,
        'num_epochs': 100,
        'num_workers': 0,
        
        # Other settings
        'early_stopping_patience': 15,
        'save_metrics': True,
        'create_plots': True,
    }
    
    print("="*60)
    print("Self-Supervised Contrastive Learning for Parkinson's Detection")
    print("="*60)
    print("\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Phase 1: Pre-training
    pretrained_encoder, pretrain_history = pretrain_self_supervised(config)
    
    # Phase 2: Fine-tuning with K-Fold
    fold_results = finetune_with_kfold(pretrained_encoder, config)
    
    return pretrained_encoder, fold_results


if __name__ == "__main__":
    encoder, results = main()
