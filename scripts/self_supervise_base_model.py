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
from scipy.signal import butter, filtfilt, resample_poly
from math import gcd as _gcd
gcd = _gcd
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

def add_gaussian_noise(data: np.ndarray, noise_level: float = 0.01) -> np.ndarray:
    """Add Gaussian noise to the signal."""
    noise = np.random.normal(0, noise_level * np.std(data), data.shape)
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
    g = gcd(original_freq, target_freq)
    up = target_freq // g
    down = original_freq // g
    return resample_poly(data, up, down, axis=0)


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


def k_fold_split_method_contrastive(data_root, full_dataset, k=5):
    """Stratified K-Fold split at patient level for ContrastiveDataset."""
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
            
    # Only include patients that are in the dataset
    dataset_pids = set(np.unique(full_dataset.patient_ids))
    
    patient_list = []
    patient_labels = []
    for pid in sorted(patient_conditions.keys()):
        if pid not in dataset_pids:
            continue
        condition = patient_conditions[pid]
        if condition == 'Healthy':
            label = 0
        elif 'Parkinson' in condition:
            label = 1
        else:
            label = 2
        patient_list.append(pid)
        patient_labels.append(label)
    
    print(f"Contrastive split — patients: {len(patient_list)} "
          f"(HC={patient_labels.count(0)}, PD={patient_labels.count(1)}, DD={patient_labels.count(2)})")
    
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    fold_datasets = []
    
    for fold_id, (train_idx, val_idx) in enumerate(skf.split(patient_list, patient_labels)):
        train_patients = set([patient_list[i] for i in train_idx])
        val_patients = set([patient_list[i] for i in val_idx])
        
        train_mask = np.array([pid in train_patients for pid in full_dataset.patient_ids])
        val_mask = np.array([pid in val_patients for pid in full_dataset.patient_ids])
        
        train_dataset = ContrastiveDataset(
            data_root=None,
            left_samples=full_dataset.left_samples[train_mask],
            right_samples=full_dataset.right_samples[train_mask],
            patient_ids=full_dataset.patient_ids[train_mask],
            class_labels=full_dataset.class_labels[train_mask],
            augmentation_strength=full_dataset.augmentation_strength,
            augmentation_type=full_dataset.augmentation_type,
            ssl_variant=full_dataset.ssl_variant
        )
        
        val_dataset = ContrastiveDataset(
            data_root=None,
            left_samples=full_dataset.left_samples[val_mask],
            right_samples=full_dataset.right_samples[val_mask],
            patient_ids=full_dataset.patient_ids[val_mask],
            class_labels=full_dataset.class_labels[val_mask],
            augmentation_strength=full_dataset.augmentation_strength,
            augmentation_type=full_dataset.augmentation_type,
            ssl_variant=full_dataset.ssl_variant
        )
        
        print(f"\n  Fold {fold_id+1}/{k}: Train={len(train_dataset)} samples, Val={len(val_dataset)} samples")
        fold_datasets.append((train_dataset, val_dataset))
    
    return fold_datasets


# ============================================================================
# CONTRASTIVE DATASET (for pre-training)
# ============================================================================

class ContrastiveDataset(Dataset):
    """
    Dataset for self-supervised contrastive pre-training.

    SSL — NTXent / InfoNCE:  anchor=aug₁, positive=aug₂ (two independent views)
    SSL — Triplet:           anchor=raw,  positive=aug
    """
    
    def __init__(
        self,
        data_root: str = None,
        window_size: int = 256,
        left_samples=None,
        right_samples=None,
        apply_dowsampling: bool = True,
        apply_bandpass_filter: bool = True,
        augmentation_strength: float = 0.3,
        augmentation_type: int = 2,  # 1=gaussian_noise, 2=time_warp, 3=both_random
        ssl_variant: str = 'ntxent', # 'ntxent'/'infonce' → two-view aug; 'triplet' → raw+aug
        **kwargs
    ):
        self.left_samples = []
        self.right_samples = []
        self.patient_ids = []
        self.task_names = []
        self.class_labels = []  # 0=HC, 1=PD, 2=DD
        self.apply_dowsampling = apply_dowsampling
        self.apply_bandpass_filter = apply_bandpass_filter
        self.augmentation_strength = augmentation_strength
        self.augmentation_type = augmentation_type
        self.ssl_variant = ssl_variant   # controls anchor/positive construction
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
            self._build_class_indices()  # Build indices for efficient class-aware sampling
        else:
            if left_samples is not None:
                self.left_samples = np.array(left_samples) if not isinstance(left_samples, np.ndarray) else left_samples
            if right_samples is not None:
                self.right_samples = np.array(right_samples) if not isinstance(right_samples, np.ndarray) else right_samples
            self.patient_ids = kwargs.get('patient_ids', [])
            if self.patient_ids is not None and len(self.patient_ids) > 0:
                self.patient_ids = np.array(self.patient_ids) if not isinstance(self.patient_ids, np.ndarray) else self.patient_ids
            self.class_labels = kwargs.get('class_labels', [])
            if self.class_labels is not None and len(self.class_labels) > 0:
                self.class_labels = np.array(self.class_labels) if not isinstance(self.class_labels, np.ndarray) else self.class_labels
                self._build_class_indices()

    def _load_data(self):
        class_counts = {'HC': 0, 'PD': 0, 'DD': 0}
        
        for patient_id in tqdm(self.patient_ids_list, desc="Loading patients (differential sampling)"):
            patient_path = pathlib.Path(str(self.patients_template).format(p=patient_id))
            
            if not patient_path.exists():
                continue
                
            try:
                with open(patient_path, 'r') as f:
                    metadata = json.load(f)
                
                condition = metadata.get('condition', '')
                
                if condition == 'Healthy':
                    overlap = 0.73  # More windows for minority class
                    class_name = 'HC'
                    class_label = 0  # HC
                elif 'Parkinson' in condition:
                    overlap = 0.0   # Fewer windows for majority class
                    class_name = 'PD'
                    class_label = 1  # PD
                else:  # DD
                    overlap = 0.65  # Moderate overlap
                    class_name = 'DD'
                    class_label = 2  # DD
                
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
                        
                        # Use class-based overlap for differential sampling
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
                    class_counts[class_name] += len(patient_left_samples)
                    
                    for i in range(len(patient_left_samples)):
                        self.left_samples.append(patient_left_samples[i])
                        self.right_samples.append(patient_right_samples[i])
                        self.patient_ids.append(patient_id)
                        self.task_names.append(patient_task_names[i])
                        self.class_labels.append(class_label)  # Store class label for negative sampling
                
            except Exception as e:
                continue
        
        # Print diagnostic info
        total_samples = len(self.left_samples)
        print(f"\nData loading summary:")
        print(f"  Data root: {self.data_root}")
        print(f"  Class counts: HC={class_counts['HC']}, PD={class_counts['PD']}, DD={class_counts['DD']}")
        print(f"  Total samples loaded: {total_samples}")
        
        self.left_samples = np.array(self.left_samples)
        self.right_samples = np.array(self.right_samples)
        self.patient_ids = np.array(self.patient_ids)
        self.task_names = np.array(self.task_names)
        self.class_labels = np.array(self.class_labels)
        
    def _build_class_indices(self):
        """Build indices for each class for efficient sampling."""
        self.hc_indices = np.where(self.class_labels == 0)[0]  # Healthy Control
        self.pd_indices = np.where(self.class_labels == 1)[0]  # Parkinson's Disease
        self.dd_indices = np.where(self.class_labels == 2)[0]  # Differential Diagnosis
        self.diseased_indices = np.concatenate([self.pd_indices, self.dd_indices])  # PD + DD
        
    def __len__(self):
        return len(self.left_samples)
    
    def _apply_augmentation(self, data, aug_type=None):
        """
        aug_type 1: Gaussian noise  (noise_level = augmentation_strength)
        aug_type 2: Time warp       (sigma       = augmentation_strength)
        aug_type 3: Random per-call coin flip — rand = np.random.random()
                    rand < 0.5 → gaussian noise (noise_level = augmentation_strength)
                    rand >= 0.5 → time warp     (sigma       = augmentation_strength)
        """
        if aug_type is None:
            aug_type = self.augmentation_type
            
        if aug_type == 1:
            return add_gaussian_noise(data, noise_level=self.augmentation_strength)
        elif aug_type == 2:
            return time_warp(data, sigma=self.augmentation_strength)
        elif aug_type == 3:
            # Independent coin flip per call — same strength regardless of which transform
            rand = np.random.random()
            if rand < 0.5:
                return add_gaussian_noise(data, noise_level=self.augmentation_strength)
            else:
                return time_warp(data, sigma=self.augmentation_strength)
        else:
            return data  # fallback: no augmentation
    
    def __getitem__(self, idx):
        """
        Returns 6 tensors: (left_anchor, right_anchor, left_positive, right_positive,
                            left_negative, right_negative)

        SSL NTXent/InfoNCE: anchor=aug₁, positive=aug₂ (two independent views)
        SSL Triplet:        anchor=raw,  positive=aug
        """
        anchor_patient = self.patient_ids[idx]

        # --- Anchor & Positive -------------------------------------------------
        if self.ssl_variant in ('ntxent', 'infonce'):
            # Two independent augmented views of the same sample
            left_anchor    = self._apply_augmentation(self.left_samples[idx].copy())
            right_anchor   = self._apply_augmentation(self.right_samples[idx].copy())
            left_positive  = self._apply_augmentation(self.left_samples[idx].copy())
            right_positive = self._apply_augmentation(self.right_samples[idx].copy())
        else:
            # Triplet: anchor=raw (clean), positive=augmented
            left_anchor    = self.left_samples[idx].copy()
            right_anchor   = self.right_samples[idx].copy()
            left_positive  = self._apply_augmentation(self.left_samples[idx].copy())
            right_positive = self._apply_augmentation(self.right_samples[idx].copy())

        # --- Negative: random sample from a different patient ------------------
        max_attempts = 50
        neg_idx = idx
        for _ in range(max_attempts):
            neg_idx = np.random.randint(0, len(self.left_samples))
            if self.patient_ids[neg_idx] != anchor_patient:
                break

        # Negative augmentation:
        #   NTXent/InfoNCE: augment negative (standard two-view practice)
        #   Triplet / aug_type==3: augment via coin flip
        if self.augmentation_type == 3:
            left_negative  = self._apply_augmentation(self.left_samples[neg_idx].copy())
            right_negative = self._apply_augmentation(self.right_samples[neg_idx].copy())
        elif self.ssl_variant in ('ntxent', 'infonce'):
            left_negative  = self._apply_augmentation(self.left_samples[neg_idx].copy())
            right_negative = self._apply_augmentation(self.right_samples[neg_idx].copy())
        else:
            left_negative  = self.left_samples[neg_idx].copy()
            right_negative = self.right_samples[neg_idx].copy()

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

    if fold_metrics_pd:
        pd_filename = f"metrics/ssl_pd_vs_dd_metrics{fold_suffix}.csv"
        write_csv(pd_filename, fold_metrics_pd)

def plot_loss(train_losses, val_losses, output_path):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss', color='blue')
    plt.plot(val_losses, label='Validation Loss', color='orange')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training and Validation Loss', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


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
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, "tsne_pd_vs_dd.png"), dpi=150, bbox_inches='tight')
        plt.close()

    return features_2d


# ============================================================================
# KNN PROBE (collapse detection during pretraining)
# ============================================================================

def knn_probe(encoder, labeled_loader, device, n_neighbors=5):
    """
    Run a k-NN classifier on encoder features to detect model collapse.
    
    Returns dict with HC-vs-PD and PD-vs-DD accuracy.
    A result near 50% on binary tasks signals collapse.
    """
    from sklearn.neighbors import KNeighborsClassifier
    
    encoder.eval()
    all_features = []
    all_hc_pd = []
    all_pd_dd = []
    
    with torch.no_grad():
        for batch in labeled_loader:
            left, right, hc_pd, pd_dd = batch[0], batch[1], batch[2], batch[3]
            left, right = left.to(device), right.to(device)
            feats = encoder.get_features(left, right)
            all_features.append(feats.cpu().numpy())
            all_hc_pd.append(hc_pd.numpy())
            all_pd_dd.append(pd_dd.numpy())
    
    X = np.concatenate(all_features)
    y_hc_pd = np.concatenate(all_hc_pd)
    y_pd_dd = np.concatenate(all_pd_dd)
    
    results = {}
    
    # HC vs PD
    valid_hc_pd = y_hc_pd >= 0
    if valid_hc_pd.sum() > n_neighbors:
        knn = KNeighborsClassifier(n_neighbors=n_neighbors)
        knn.fit(X[valid_hc_pd], y_hc_pd[valid_hc_pd])
        results['acc_hc_pd'] = knn.score(X[valid_hc_pd], y_hc_pd[valid_hc_pd])
    
    # PD vs DD
    valid_pd_dd = y_pd_dd >= 0
    if valid_pd_dd.sum() > n_neighbors:
        knn = KNeighborsClassifier(n_neighbors=n_neighbors)
        knn.fit(X[valid_pd_dd], y_pd_dd[valid_pd_dd])
        results['acc_pd_dd'] = knn.score(X[valid_pd_dd], y_pd_dd[valid_pd_dd])
    
    encoder.train()
    return results


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def train_contrastive_epoch(model, dataloader, criterion, optimizer, device):
    """Train encoder for one epoch with SSL contrastive loss."""
    model.train()
    total_loss = 0.0

    for batch in tqdm(dataloader, desc="Contrastive Training"):
        (left_anchor, right_anchor,
         left_positive, right_positive,
         left_negative, right_negative) = batch

        left_anchor    = left_anchor.to(device)
        right_anchor   = right_anchor.to(device)
        left_positive  = left_positive.to(device)
        right_positive = right_positive.to(device)
        left_negative  = left_negative.to(device)
        right_negative = right_negative.to(device)

        optimizer.zero_grad()

        anchor_emb   = model(left_anchor,   right_anchor)
        positive_emb = model(left_positive, right_positive)
        negative_emb = model(left_negative, right_negative)

        loss = criterion(anchor_emb, positive_emb, negative_emb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def validate_contrastive_epoch(model, dataloader, criterion, device):
    """Validate encoder for one epoch with SSL contrastive loss (no gradients)."""
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Contrastive Validation"):
            (left_anchor, right_anchor,
             left_positive, right_positive,
             left_negative, right_negative) = batch

            left_anchor    = left_anchor.to(device)
            right_anchor   = right_anchor.to(device)
            left_positive  = left_positive.to(device)
            right_positive = right_positive.to(device)
            left_negative  = left_negative.to(device)
            right_negative = right_negative.to(device)

            anchor_emb   = model(left_anchor,   right_anchor)
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
            
            features = model.get_features(left_sample, right_sample)
            
            all_features.append(features.cpu().numpy())
            all_hc_pd_labels.append(hc_pd.numpy())
            all_pd_dd_labels.append(pd_dd.numpy())
    
    all_features = np.vstack(all_features)
    all_hc_pd_labels = np.concatenate(all_hc_pd_labels)
    all_pd_dd_labels = np.concatenate(all_pd_dd_labels)
    
    return all_features, all_hc_pd_labels, all_pd_dd_labels



def transfer_weights_to_classifier(pretrained_encoder, classifier_model):
    classifier_model.left_projection.load_state_dict(pretrained_encoder.left_projection.state_dict())
    classifier_model.right_projection.load_state_dict(pretrained_encoder.right_projection.state_dict())
    classifier_model.positional_encoding.load_state_dict(pretrained_encoder.positional_encoding.state_dict())
    
    for pretrained_layer, classifier_layer in zip(pretrained_encoder.layers, classifier_model.layers):
        classifier_layer.load_state_dict(pretrained_layer.state_dict())
    
    print("✓ Transferred pre-trained weights to classifier model")
    return classifier_model


def load_pretrained_encoder(checkpoint_path, config, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Loading pre-trained encoder from: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Use config from checkpoint if available, otherwise use provided config
    saved_config = checkpoint.get('config', config)
    
    # Create model with same architecture
    model = ContrastiveEncoder(
        input_dim=saved_config.get('input_dim', config.get('input_dim', 6)),
        model_dim=saved_config.get('model_dim', config.get('model_dim', 32)),
        num_heads=saved_config.get('num_heads', config.get('num_heads', 8)),
        num_layers=saved_config.get('num_layers', config.get('num_layers', 3)),
        d_ff=saved_config.get('d_ff', config.get('d_ff', 256)),
        dropout=saved_config.get('dropout', config.get('dropout', 0.1)),
        timestep=saved_config.get('timestep', config.get('timestep', 256)),
        projection_dim=saved_config.get('projection_dim', config.get('projection_dim', 64))
    ).to(device)
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    
    epoch_info = checkpoint.get('epoch', 'N/A')
    val_loss = checkpoint.get('val_loss', 'N/A')
    val_loss_str = f"{val_loss:.4f}" if isinstance(val_loss, float) else str(val_loss)
    
    print(f"✓ Loaded pre-trained encoder (epoch: {epoch_info}, val_loss: {val_loss_str})")
    
    return model, checkpoint


def load_finetuned_classifier(checkpoint_path, config, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Loading fine-tuned classifier from: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Use config from checkpoint if available, otherwise use provided config
    saved_config = checkpoint.get('config', config)
    
    # Create model with same architecture
    model = MainModel(
        input_dim=saved_config.get('input_dim', config.get('input_dim', 6)),
        model_dim=saved_config.get('model_dim', config.get('model_dim', 32)),
        num_heads=saved_config.get('num_heads', config.get('num_heads', 8)),
        num_layers=saved_config.get('num_layers', config.get('num_layers', 3)),
        d_ff=saved_config.get('d_ff', config.get('d_ff', 256)),
        dropout=saved_config.get('dropout', config.get('dropout', 0.1)),
        timestep=saved_config.get('timestep', config.get('timestep', 256)),
        num_classes=saved_config.get('num_classes', config.get('num_classes', 2))
    ).to(device)
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    
    fold_info = checkpoint.get('fold', 'N/A')
    epoch_info = checkpoint.get('epoch', 'N/A')
    val_acc_combined = checkpoint.get('val_acc_combined', 'N/A')
    val_acc_hc = checkpoint.get('val_acc_hc', 'N/A')
    val_acc_pd = checkpoint.get('val_acc_pd', 'N/A')
    
    print(f"✓ Loaded fine-tuned classifier:")
    print(f"  - Fold: {fold_info}")
    print(f"  - Epoch: {epoch_info}")
    if isinstance(val_acc_combined, float):
        print(f"  - Val Acc Combined: {val_acc_combined:.4f}")
    if isinstance(val_acc_hc, float):
        print(f"  - Val Acc HC vs PD: {val_acc_hc:.4f}")
    if isinstance(val_acc_pd, float):
        print(f"  - Val Acc PD vs DD: {val_acc_pd:.4f}")
    
    return model, checkpoint


# ============================================================================
# MAIN TRAINING PIPELINE
# ============================================================================

def pretrain_self_supervised(config, exclude_patient_ids=None, fold_idx=0):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    fold_dir = f"checkpoints/fold_{fold_idx}"
    plot_pretrain_dir = f"plots/pretrain/fold_{fold_idx}"
    os.makedirs(fold_dir, exist_ok=True)
    os.makedirs(plot_pretrain_dir, exist_ok=True)

    # Save config.json once before pre-training
    config_serializable = {k: v for k, v in config.items()
                           if isinstance(v, (str, int, float, bool, list, dict, type(None)))}
    with open(f'{fold_dir}/config.json', 'w') as f:
        json.dump(config_serializable, f, indent=2)
    print(f"\u2713 Config saved: {fold_dir}/config.json")

    # --- Build dataset --------------------------------------------------------
    # ssl_variant controls augmentation strategy:
    #   'ntxent'/'infonce' → anchor=aug₁, positive=aug₂ (two-view)
    #   'triplet'          → anchor=raw,  positive=aug  (raw+aug pair)
    ssl_variant = config.get('loss_type', 'ntxent')

    print(f"\nPHASE 1: Self-Supervised Pre-training (loss: {ssl_variant.upper()})")

    full_dataset = ContrastiveDataset(
        data_root=config['data_root'],
        window_size=config.get('window_size', 256),
        apply_dowsampling=config.get('apply_downsampling', True),
        apply_bandpass_filter=config.get('apply_bandpass_filter', True),
        augmentation_strength=config.get('augmentation_strength', 0.3),
        augmentation_type=config.get('augmentation_type', 2),
        ssl_variant=ssl_variant
    )
    
    # Exclude test-fold patients from pretraining
    if exclude_patient_ids is not None and len(exclude_patient_ids) > 0:
        keep_mask = ~np.isin(full_dataset.patient_ids, exclude_patient_ids)
        full_dataset.left_samples   = full_dataset.left_samples[keep_mask]
        full_dataset.right_samples  = full_dataset.right_samples[keep_mask]
        full_dataset.patient_ids    = full_dataset.patient_ids[keep_mask]
        full_dataset.task_names     = full_dataset.task_names[keep_mask]
        full_dataset.class_labels   = full_dataset.class_labels[keep_mask]
        full_dataset._build_class_indices()
        print(f"  Excluded {len(exclude_patient_ids)} test-fold patients from pretraining")
    
    print(f"Total samples after test-fold exclusion: {len(full_dataset)}")
    
    # --- Split into train/val for pretraining (stratified at patient level) ----
    pretrain_folds = k_fold_split_method_contrastive(
        config['data_root'], full_dataset, k=config.get('num_folds', 5)
    )
    pretrain_train_dataset, pretrain_val_dataset = pretrain_folds[fold_idx % len(pretrain_folds)]
    
    print(f"\nPre-training train samples: {len(pretrain_train_dataset)}")
    print(f"Pre-training val samples:   {len(pretrain_val_dataset)}")
    
    train_loader = DataLoader(
        pretrain_train_dataset,
        batch_size=config['pretrain_batch_size'],
        shuffle=True,
        num_workers=config.get('num_workers', 0)
    )
    val_loader = DataLoader(
        pretrain_val_dataset,
        batch_size=config['pretrain_batch_size'],
        shuffle=False,
        num_workers=config.get('num_workers', 0)
    )
    
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
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.get('pretrain_lr', 1e-4),
        weight_decay=config.get('weight_decay', 1e-4)
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config['pretrain_epochs'], eta_min=1e-6
    )
    
    # --- Loss selection -------------------------------------------------------
    loss_type = config.get('loss_type', 'ntxent')
    if loss_type == 'contrastive':
        criterion = ContrastiveLoss(margin=config.get('margin', 0.3))
    elif loss_type == 'triplet':
        criterion = TripletLoss(margin=config.get('margin', 0.3))
    elif loss_type == 'infonce':
        criterion = InfoNCELoss(temperature=config.get('temperature', 0.07))
    else:  # default: ntxent
        criterion = NTXentLoss(temperature=config.get('temperature', 0.07))

    print(f"Using {loss_type} loss")
    
    # --- Labeled loader for KNN probe (validation patients only) --------------
    knn_interval = config.get('knn_probe_interval', 10)
    knn_loader = None
    if knn_interval > 0:
        # Build a labeled dataset from validation patients only for KNN probing
        val_patient_ids = set(np.unique(pretrain_val_dataset.patient_ids))
        probe_full_dataset = ParkinsonsDataLoader(
            config['data_root'],
            apply_dowsampling=config.get('apply_downsampling', True),
            apply_bandpass_filter=config.get('apply_bandpass_filter', True)
        )
        # Filter to only validation patients
        val_probe_mask = np.array([pid in val_patient_ids for pid in probe_full_dataset.patient_ids])
        probe_dataset = ParkinsonsDataLoader(
            data_root=None,
            left_samples=probe_full_dataset.left_samples[val_probe_mask],
            right_samples=probe_full_dataset.right_samples[val_probe_mask],
            hc_vs_pd=probe_full_dataset.hc_vs_pd[val_probe_mask],
            pd_vs_dd=probe_full_dataset.pd_vs_dd[val_probe_mask],
            patient_ids=probe_full_dataset.patient_ids[val_probe_mask]
        )
        print(f"KNN probe dataset (val only): {len(probe_dataset)} samples "
              f"from {len(val_patient_ids)} patients")
        knn_loader = DataLoader(
            probe_dataset, batch_size=config.get('batch_size', 32),
            shuffle=False, num_workers=config.get('num_workers', 0)
        )
    
    # --- Training loop --------------------------------------------------------
    history = {'train_loss': [], 'val_loss': [], 'knn_hc_pd': [], 'knn_pd_dd': []}
    best_val_loss = float('inf')
    patience_counter = 0
    patience = config.get('early_stopping_patience', 15)
    
    for epoch in range(config['pretrain_epochs']):
        print(f"\nPre-train Epoch {epoch + 1}/{config['pretrain_epochs']}")
        
        train_loss = train_contrastive_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss = validate_contrastive_epoch(
            model, val_loader, criterion, device
        )
        scheduler.step()
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")
        
        # KNN probe every N epochs — lightweight collapse detector (no t-SNE here)
        if knn_loader is not None and (epoch + 1) % knn_interval == 0:
            knn_results = knn_probe(model, knn_loader, device)
            hc_pd_acc = knn_results.get('acc_hc_pd', 0)
            pd_dd_acc = knn_results.get('acc_pd_dd', 0)
            history['knn_hc_pd'].append(hc_pd_acc)
            history['knn_pd_dd'].append(pd_dd_acc)
            print(f"  KNN Probe (val) → HC-PD: {hc_pd_acc:.4f}  PD-DD: {pd_dd_acc:.4f}")
            if hc_pd_acc < 0.55 and pd_dd_acc < 0.55:
                print("  ⚠ WARNING: KNN accuracy near random — possible model collapse!")
            # NOTE: t-SNE snapshots are O(N²) and run only once at end of pretraining
        
        # Early stopping based on val_loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'loss_type': ssl_variant,
                'config': config
            }, f'{fold_dir}/best_contrastive_model.pth')
            print(f"✓ Saved best pre-trained model (val_loss: {val_loss:.4f})")

        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch + 1}")
                break

    # --- Post-training plots --------------------------------------------------
    plot_loss(history['train_loss'], history['val_loss'],
              f'{plot_pretrain_dir}/contrastive_loss.png')

    if history['knn_hc_pd']:
        plt.figure(figsize=(8, 5))
        knn_epochs = list(range(knn_interval, len(history['knn_hc_pd']) * knn_interval + 1, knn_interval))
        plt.plot(knn_epochs, history['knn_hc_pd'], 'o-', label='HC vs PD', color='blue')
        plt.plot(knn_epochs, history['knn_pd_dd'], 's-', label='PD vs DD', color='red')
        plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Random chance')
        plt.xlabel('Epoch'); plt.ylabel('KNN Accuracy')
        plt.title(f'KNN Probe During SSL Pre-training [{ssl_variant.upper()}] (Validation Set)')
        plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
        plt.savefig(f'{plot_pretrain_dir}/knn_probe.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("✓ KNN probe plot saved")

    # Load best model
    checkpoint = torch.load(f'{fold_dir}/best_contrastive_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])

    # t-SNE after pretraining (on validation set)
    if config.get('create_plots', True) and knn_loader is not None:
        print("\nGenerating post-pretraining t-SNE (validation set)...")
        features, hc_pd_labels, pd_dd_labels = extract_features(
            model, knn_loader, device
        )
        if features is not None:
            plot_tsne(features, hc_pd_labels, pd_dd_labels, output_dir=plot_pretrain_dir)
            print(f"✓ t-SNE saved to {plot_pretrain_dir}/")
    
    print(f"\n✓ Pre-training complete! Best val_loss: {best_val_loss:.4f}")
    
    return model, history


def plot_label_efficiency(results, output_path):
    pcts = [r['pct'] for r in results]
    acc_hc = [r['best_val_acc_hc'] for r in results]
    acc_pd = [r['best_val_acc_pd'] for r in results]
    acc_combined = [r['best_val_acc_combined'] for r in results]
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    ax.plot(pcts, acc_hc, 'o-', color='#2196F3', linewidth=2, markersize=8, label='HC vs PD')
    ax.plot(pcts, acc_pd, 's-', color='#FF5722', linewidth=2, markersize=8, label='PD vs DD')
    ax.plot(pcts, acc_combined, 'D-', color='#4CAF50', linewidth=2.5, markersize=9, label='Combined (Avg)')
    
    ax.set_xlabel('Percentage of Labeled Training Data (%)', fontsize=13)
    ax.set_ylabel('Best Validation Accuracy', fontsize=13)
    ax.set_title('Label Efficiency Experiment', fontsize=14, fontweight='bold')
    ax.set_xticks(pcts)
    ax.set_xticklabels([f'{p}%' for p in pcts])
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])
    
    # Annotate each point with its value
    for i, pct in enumerate(pcts):
        ax.annotate(f'{acc_combined[i]:.3f}', (pct, acc_combined[i]),
                    textcoords='offset points', xytext=(0, 12), ha='center', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\u2713 Saved label efficiency plot to {output_path}")


def label_efficiency_experiment(pretrained_encoder, config, fold_idx=0):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    fold_dir = f"checkpoints/fold_{fold_idx}"
    fold_plot_dir = f"plots/label_efficiency/fold_{fold_idx}"
    os.makedirs('metrics', exist_ok=True)
    os.makedirs(fold_dir, exist_ok=True)
    os.makedirs(fold_plot_dir, exist_ok=True)

    # Save config.json once before label-efficiency training
    config_serializable = {k: v for k, v in config.items()
                           if isinstance(v, (str, int, float, bool, list, dict, type(None)))}
    with open(f'{fold_dir}/label_efficiency_config.json', 'w') as f:
        json.dump(config_serializable, f, indent=2)
    print(f"\u2713 Config saved: {fold_dir}/label_efficiency_config.json")
    
    fractions = config.get('label_efficiency_fractions', [0.20, 0.50, 0.70, 1.0])
    evaluation_type = config.get('evaluation_type', 'finetune')
    
    print(f"Fractions: {[f'{f*100:.0f}%' for f in fractions]}")
    print(f"Evaluation type: {evaluation_type}")

    full_dataset = ParkinsonsDataLoader(
        config['data_root'],
        apply_dowsampling=config['apply_downsampling'],
        apply_bandpass_filter=config['apply_bandpass_filter']
    )
    fold_datasets = full_dataset.get_train_test_split(split_type=3, k=config['num_folds'])

    train_dataset, val_dataset = fold_datasets[fold_idx]

    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'],
                            shuffle=False, num_workers=config['num_workers'])

    n_train = len(train_dataset)
    print(f"\nFold {fold_idx+1}: Total training samples = {n_train}, Validation samples = {len(val_dataset)}")

    permutation_path = f'metrics/label_efficiency_permutation_fold{fold_idx}.json'
    
    if os.path.exists(permutation_path):
        with open(permutation_path, 'r') as f:
            perm_data = json.load(f)
        permuted_indices = np.array(perm_data['permuted_indices'])
        completed_fractions = set(perm_data.get('completed_fractions', []))
        all_fraction_results = perm_data.get('completed_results', [])
        
        print(f"✓ Loaded existing permutation from {permutation_path}")
        print(f"  Total indices: {len(permuted_indices)}")
        print(f"  Already completed fractions: {sorted(completed_fractions) if completed_fractions else 'None'}")
        
        # Verify consistency
        if len(permuted_indices) != n_train:
            print(f"⚠ WARNING: Saved permutation size ({len(permuted_indices)}) != training set size ({n_train}). Regenerating...")
            os.remove(permutation_path)
            permuted_indices = None
    else:
        permuted_indices = None
        completed_fractions = set()
        all_fraction_results = []
    
    if permuted_indices is None:
        rng = np.random.RandomState(42)
        
        hc_pd_labels = np.array(train_dataset.hc_vs_pd)
        pd_dd_labels = np.array(train_dataset.pd_vs_dd)
        
        combined_labels = np.zeros(n_train, dtype=int)
        combined_labels[hc_pd_labels == 0] = 0                           # HC
        combined_labels[(hc_pd_labels == 1) & (pd_dd_labels == 0)] = 1   # PD
        combined_labels[pd_dd_labels == 1] = 2                           # DD
        
        ordered_indices = []
        for cls in np.unique(combined_labels):
            cls_indices = np.where(combined_labels == cls)[0]
            rng.shuffle(cls_indices)
            ordered_indices.append(cls_indices)
        
        max_len = max(len(idx) for idx in ordered_indices)
        interleaved = []
        for i in range(max_len):
            for cls_indices in ordered_indices:
                if i < len(cls_indices):
                    interleaved.append(cls_indices[i])
        permuted_indices = np.array(interleaved)
        
        # Build fraction boundaries for documentation
        fraction_info = {}
        for frac in fractions:
            n_sub = min(int(np.ceil(frac * n_train)), n_train)
            pct_key = int(round(frac * 100))
            fraction_info[f"{pct_key}%"] = {'n_samples': n_sub, 'start_idx': 0, 'end_idx': n_sub}
        
        perm_data = {
            'n_train': n_train,
            'n_val': len(val_dataset),
            'fold_idx': fold_idx,
            'random_seed': 42,
            'fractions': [float(f) for f in fractions],
            'fraction_boundaries': fraction_info,
            'permuted_indices': permuted_indices.tolist(),
            'completed_fractions': [],
            'completed_results': []
        }
        with open(permutation_path, 'w') as f:
            json.dump(perm_data, f, indent=2)
        
        print(f"✓ Saved new permutation to {permutation_path}")
        print(f"  Fraction boundaries: {fraction_info}")
        
        completed_fractions = set()
        all_fraction_results = []
    
    for frac in fractions:
        n_subset = int(np.ceil(frac * n_train))
        n_subset = min(n_subset, n_train)  # cap at total
        subset_indices = permuted_indices[:n_subset] if n_subset > 0 else np.array([], dtype=int)
        
        pct = int(round(frac * 100))
        
        # Skip if this fraction was already completed in a previous run
        if pct in completed_fractions:
            print(f"\n{'='*60}")
            print(f"Skipping {pct}% — already completed in a previous run")
            print(f"{'='*60}")
            continue
        
        hc_pd_loss = nn.CrossEntropyLoss()
        pd_dd_loss  = nn.CrossEntropyLoss()
        
        # Create subset dataset
        subset_dataset = ParkinsonsDataLoader(
            data_root=None,
            left_samples=train_dataset.left_samples[subset_indices],
            right_samples=train_dataset.right_samples[subset_indices],
            hc_vs_pd=train_dataset.hc_vs_pd[subset_indices],
            pd_vs_dd=train_dataset.pd_vs_dd[subset_indices],
            patient_ids=train_dataset.patient_ids[subset_indices]
        )
        
        train_loader = DataLoader(subset_dataset, batch_size=config['batch_size'],
                                  shuffle=True, num_workers=config['num_workers'])
        
        # Class distribution of the subset
        n_hc = int(np.sum(subset_dataset.hc_vs_pd == 0))
        n_pd = int(np.sum((subset_dataset.hc_vs_pd == 1) & (subset_dataset.pd_vs_dd == 0)))
        n_dd = int(np.sum(subset_dataset.pd_vs_dd == 1))
        
        print(f"\n{'='*60}")
        print(f"Training with {pct}% of labels ({n_subset}/{n_train} samples)")
        print(f"  Class distribution: HC={n_hc}, PD={n_pd}, DD={n_dd}")
        print(f"{'='*60}")
        
        # Fresh model for each fraction (fair comparison)
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
        
        # Freeze or unfreeze based on evaluation type
        if evaluation_type == 'linear':
            for param in model.left_projection.parameters():
                param.requires_grad = False
            for param in model.right_projection.parameters():
                param.requires_grad = False
            for param in model.positional_encoding.parameters():
                param.requires_grad = False
            for layer in model.layers:
                for param in layer.parameters():
                    param.requires_grad = False
            trainable_params = list(model.head_hc_vs_pd.parameters()) + list(model.head_pd_vs_dd.parameters())
            optimizer = optim.AdamW(trainable_params, lr=config['learning_rate'], weight_decay=config['weight_decay'])
        else:
            optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        
        best_val_acc = 0.0
        best_val_acc_hc = 0.0
        best_val_acc_pd = 0.0
        best_epoch = 0
        best_val_cse = float('inf')
        best_metrics_hc = {}
        best_metrics_pd = {}
        patience_counter = 0
        patience = config.get('early_stopping_patience', 15)
        epoch_history = []  # per-epoch tracking for 100% fraction
        
        for epoch in range(config['num_epochs']):
            train_loss, train_metrics_hc, train_metrics_pd = train_single_epoch(
                model, train_loader, hc_pd_loss, pd_dd_loss, optimizer, device
            )
            
            val_results = validate_single_epoch(
                model, val_loader, hc_pd_loss, pd_dd_loss, device
            )
            val_loss, hc_pd_val_pred, hc_pd_val_labels, hc_pd_val_probs, \
            pd_dd_val_pred, pd_dd_val_labels, pd_dd_val_probs = val_results
            
            val_metrics_hc = calculate_metrics(
                hc_pd_val_labels, hc_pd_val_pred,
                f"{pct}% Validation HC vs PD", verbose=False
            )
            val_metrics_pd = calculate_metrics(
                pd_dd_val_labels, pd_dd_val_pred,
                f"{pct}% Validation PD vs DD", verbose=False
            )
            
            val_acc_hc = val_metrics_hc.get('accuracy', 0)
            val_acc_pd = val_metrics_pd.get('accuracy', 0)
            val_acc_combined = (val_acc_hc + val_acc_pd) / 2
            
            scheduler.step(val_loss)
            
            epoch_history.append({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_acc_hc': val_acc_hc,
                'val_acc_pd': val_acc_pd,
                'val_acc_combined': val_acc_combined
            })
            
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"  Epoch {epoch+1}/{config['num_epochs']} | "
                      f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                      f"Val Acc HC: {val_acc_hc:.4f} PD: {val_acc_pd:.4f} Combined: {val_acc_combined:.4f}")
            
            if val_acc_combined > best_val_acc:
                best_val_acc = val_acc_combined
                best_val_acc_hc = val_acc_hc
                best_val_acc_pd = val_acc_pd
                best_epoch = epoch + 1
                best_val_cse = val_loss
                best_metrics_hc = val_metrics_hc
                best_metrics_pd = val_metrics_pd
                patience_counter = 0
                
                # Save model checkpoint with config
                checkpoint_path = f'{fold_dir}/label_eff_{pct}pct_best.pth'
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'fraction': frac,
                    'pct': pct,
                    'epoch': epoch,
                    'val_acc_combined': val_acc_combined,
                    'val_acc_hc': val_acc_hc,
                    'val_acc_pd': val_acc_pd,
                    'config': config
                }, checkpoint_path)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"  Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
                    break
        
        result = {
            'fraction': frac,
            'pct': pct,
            'n_samples': n_subset,
            'best_val_acc_hc': best_val_acc_hc,
            'best_val_acc_pd': best_val_acc_pd,
            'best_val_acc_combined': best_val_acc,
            'precision_hc': best_metrics_hc.get('precision_avg', 0),
            'recall_hc':    best_metrics_hc.get('recall_avg', 0),
            'f1_hc':        best_metrics_hc.get('f1_avg', 0),
            'precision_pd': best_metrics_pd.get('precision_avg', 0),
            'recall_pd':    best_metrics_pd.get('recall_avg', 0),
            'f1_pd':        best_metrics_pd.get('f1_avg', 0),
            'val_cse':      best_val_cse,
            'best_epoch': best_epoch
        }
        all_fraction_results.append(result)
    
        if frac == 1.0 and config.get('create_plots', True):
            print("\n[100%] Generating full evaluation artefacts...")
            full_plot_dir = f'{fold_plot_dir}/full_finetune'
            os.makedirs(full_plot_dir, exist_ok=True)
            
            # 1. Per-epoch metric CSV
            epoch_csv_path = f'metrics/label_efficiency_100pct_epochs_fold{fold_idx}.csv'
            with open(epoch_csv_path, 'w', newline='') as ef:
                writer = csv.DictWriter(ef, fieldnames=[
                    'epoch', 'train_loss', 'val_loss',
                    'val_acc_hc', 'val_acc_pd', 'val_acc_combined'
                ])
                writer.writeheader()
                for ep_data in epoch_history:
                    writer.writerow(ep_data)
            print(f"  \u2713 Per-epoch CSV saved to {epoch_csv_path}")
            
            # 2. Training curves — loss via plot_loss, accuracy inline
            plot_loss(
                [e['train_loss'] for e in epoch_history],
                [e['val_loss']   for e in epoch_history],
                f'{full_plot_dir}/loss_curve.png'
            )
            
            plt.figure(figsize=(8, 5))
            plt.plot([e['val_acc_combined'] for e in epoch_history], label='Combined', color='green')
            plt.plot([e['val_acc_hc'] for e in epoch_history], label='HC vs PD', color='blue', linestyle='--')
            plt.plot([e['val_acc_pd'] for e in epoch_history], label='PD vs DD', color='red', linestyle='--')
            plt.xlabel('Epoch', fontsize=12); plt.ylabel('Accuracy', fontsize=12)
            plt.title('100% Fine-Tuning — Validation Accuracy', fontsize=13)
            plt.legend(fontsize=10); plt.grid(alpha=0.3); plt.tight_layout()
            plt.savefig(f'{full_plot_dir}/accuracy_curve.png', dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  \u2713 Training curves saved (loss_curve.png, accuracy_curve.png)")
            
            # 3. ROC Curves (use last val epoch probs)
            val_results_full = validate_single_epoch(model, val_loader, hc_pd_loss, pd_dd_loss, device)
            _, hc_pd_p, hc_pd_l, hc_pd_probs, pd_dd_p, pd_dd_l, pd_dd_probs = val_results_full
            if hc_pd_probs:
                plot_roc_curves(np.array(hc_pd_l), np.array(hc_pd_p), np.array(hc_pd_probs),
                                f'{full_plot_dir}/roc_hc_vs_pd.png')
                print(f"  \u2713 ROC curve (HC vs PD) saved")
            if pd_dd_probs:
                plot_roc_curves(np.array(pd_dd_l), np.array(pd_dd_p), np.array(pd_dd_probs),
                                f'{full_plot_dir}/roc_pd_vs_dd.png')
                print(f"  \u2713 ROC curve (PD vs DD) saved")
            
            # 4. t-SNE
            features, hc_pd_feat_labels, pd_dd_feat_labels = extract_features(model, val_loader, device)
            if features is not None:
                plot_tsne(features, hc_pd_feat_labels, pd_dd_feat_labels, output_dir=full_plot_dir)
                print(f"  \u2713 t-SNE plots saved")
            
            print(f"  \u2713 Full 100% evaluation complete, outputs in {full_plot_dir}/")

        
        # Mark this fraction as completed and save progress
        completed_fractions.add(pct)
        with open(permutation_path, 'r') as f:
            perm_data = json.load(f)
        perm_data['completed_fractions'] = sorted(list(completed_fractions))
        perm_data['completed_results'] = all_fraction_results
        with open(permutation_path, 'w') as f:
            json.dump(perm_data, f, indent=2)
        
        print(f"\n✓ {pct}% labels: Best Combined Acc = {best_val_acc:.4f} "
              f"(HC: {best_val_acc_hc:.4f}, PD: {best_val_acc_pd:.4f}) at epoch {best_epoch}")
        print(f"  Progress saved to {permutation_path}")
    

    # Save results to CSV
    csv_path = f'metrics/label_efficiency_results_fold{fold_idx}.csv'
    with open(csv_path, 'w') as f:
        f.write('pct_labels,n_samples,acc_hc_vs_pd,acc_pd_vs_dd,acc_combined,'
                'precision_hc,recall_hc,f1_hc,precision_pd,recall_pd,f1_pd,val_cse,best_epoch\n')
        for r in all_fraction_results:
            f.write(
                f"{r['pct']},{r['n_samples']},"
                f"{r['best_val_acc_hc']:.4f},{r['best_val_acc_pd']:.4f},{r['best_val_acc_combined']:.4f},"
                f"{r.get('precision_hc', 0):.4f},{r.get('recall_hc', 0):.4f},{r.get('f1_hc', 0):.4f},"
                f"{r.get('precision_pd', 0):.4f},{r.get('recall_pd', 0):.4f},{r.get('f1_pd', 0):.4f},"
                f"{r.get('val_cse', 0):.4f},{r['best_epoch']}\n"
            )
    print(f"\n\u2713 Saved results to {csv_path}")
    
    # Plot accuracy vs label percentage
    plot_label_efficiency(all_fraction_results, f'{fold_plot_dir}/accuracy_vs_labels.png')

    return all_fraction_results


def main():
    """Main function for contrastive learning pipeline."""

    USE_NESTED_CV = False  
    RESULTS_DIR = "/kaggle/input/notebooks/meheruzannat/parkinsons-ssl"      

    base_config = {
        # Data settings
        'data_root': "/kaggle/input/datasets/meherujannat/parkinsons/pads-parkinsons-disease-smartwatch-dataset-1.0.0",
        'apply_downsampling': True,
        'apply_bandpass_filter': True,
        'window_size': 256,
        'augmentation_strength': 0.3,
        'augmentation_type': 3,  # 1=gaussian_noise, 2=time_warp, 3=both_random

        'input_dim': 6,
        'timestep': 256,
        'num_classes': 2,
        'projection_dim': 64,
        'load_pretrained_checkpoint': None,
        'load_finetuned_checkpoint': None,

        # Pre-training settings
        'pretrain_batch_size': 64,
        'pretrain_lr': 1e-4,
        'pretrain_epochs': 50,
        # SSL loss type — also controls augmentation strategy:
        #   'ntxent' / 'infonce' → anchor=aug₁, positive=aug₂ (two-view)
        #   'triplet'            → anchor=raw,  positive=aug  (raw+aug pair)
        'loss_type': 'infonce',
        'margin': 0.3,
        'temperature': 0.07,
        'knn_probe_interval': 0,  # KNN probe every N epochs (0 to disable)

        # Fine-tuning / Linear Probing settings
        'evaluation_type': 'finetune',  # 'linear' or 'finetune'
        'num_folds': 5,
        'num_epochs': 50,
        'num_workers': 0,

        # Other settings
        'early_stopping_patience': 10,
        'save_metrics': True,
        'create_plots': True,

        # Label Efficiency Experiment (100% acts as full fine-tuning)
        'label_efficiency_fractions': [0.05, 0.10, 0.20, 0.50, 0.70, 1.0],
    }

    # ---------- initial cross-validation hyperparameters ----------
    initial_cv_config = {
        'model_dim': 64,
        'num_heads': 8,
        'num_layers': 3,
        'd_ff': 256,
        'dropout': 0.2,
        'batch_size': 32,
        'learning_rate': 0.0005,
        'weight_decay': 0.01,
    }

    # ---------- nested cross-validation hyperparameters ----------
    nested_cv_config = {
        'model_dim': 32,
        'num_heads': 8,
        'num_layers': 3,
        'd_ff': 256,
        'dropout': 0.12281570220908891,
        'batch_size': 32,
        'learning_rate': 0.0002912623775216651,
        'weight_decay': 0.00016228510005606125,
    }

    # ---------- merge ----------
    config = {**base_config, **(nested_cv_config if USE_NESTED_CV else initial_cv_config)}
    config['hp_source'] = 'nested_cv' if USE_NESTED_CV else 'initial_cv'

    print(f"Using {'nested CV' if USE_NESTED_CV else 'initial CV'} hyperparameters")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_folds = config['num_folds']

    if RESULTS_DIR is not None:
        import shutil
        results_path = pathlib.Path(RESULTS_DIR)
        for subdir in ['checkpoints', 'metrics', 'plots']:
            src = results_path / subdir
            if src.exists():
                dst = pathlib.Path(subdir)
                if src.is_dir():
                    for item in src.rglob('*'):
                        if item.is_file():
                            dest_file = dst / item.relative_to(src)
                            dest_file.parent.mkdir(parents=True, exist_ok=True)
                            if not dest_file.exists():
                                shutil.copy2(str(item), str(dest_file))
                                print(f"  Restored: {dest_file}")
        print(f"✓ Restored prior results from {RESULTS_DIR}")

        # ── Migrate legacy FLAT structure → fold-indexed structure ───────────
        # Old saved notebooks had no fold subdirs. Map them to fold_0 so the
        # fold-completion checks and pretrain checkpoint checks find them.
        import shutil as _shutil

        _legacy_pcts = [5, 10, 20, 50, 70, 100]

        # 1. Pretrain checkpoint
        flat_pretrain = pathlib.Path('checkpoints/best_contrastive_model.pth')
        fold0_pretrain = pathlib.Path('checkpoints/fold_0/best_contrastive_model.pth')
        if flat_pretrain.exists() and not fold0_pretrain.exists():
            fold0_pretrain.parent.mkdir(parents=True, exist_ok=True)
            _shutil.copy2(str(flat_pretrain), str(fold0_pretrain))
            print(f"  ↳ Migrated flat pretrain ckpt → {fold0_pretrain}")

        # 2. Label-efficiency checkpoints (all fractions)
        for _pct in _legacy_pcts:
            flat_ckpt = pathlib.Path(f'checkpoints/label_eff_{_pct}pct_best.pth')
            fold_ckpt = pathlib.Path(f'checkpoints/fold_0/label_eff_{_pct}pct_best.pth')
            if flat_ckpt.exists() and not fold_ckpt.exists():
                fold_ckpt.parent.mkdir(parents=True, exist_ok=True)
                _shutil.copy2(str(flat_ckpt), str(fold_ckpt))
                print(f"  ↳ Migrated flat {_pct}% ckpt → {fold_ckpt}")

        # 3. Permutation JSON  (controls fold-skip logic)
        flat_perm = pathlib.Path('metrics/label_efficiency_permutation.json')
        fold0_perm = pathlib.Path('metrics/label_efficiency_permutation_fold0.json')
        if flat_perm.exists() and not fold0_perm.exists():
            _shutil.copy2(str(flat_perm), str(fold0_perm))
            print(f"  ↳ Migrated flat permutation JSON → {fold0_perm}")

        # 4. Results CSV
        flat_res = pathlib.Path('metrics/label_efficiency_results.csv')
        fold0_res = pathlib.Path('metrics/label_efficiency_results_fold0.csv')
        if flat_res.exists() and not fold0_res.exists():
            _shutil.copy2(str(flat_res), str(fold0_res))
            print(f"  ↳ Migrated flat results CSV → {fold0_res}")

        # 5. Per-epoch 100% CSV
        flat_ep = pathlib.Path('metrics/label_efficiency_100pct_epochs.csv')
        fold0_ep = pathlib.Path('metrics/label_efficiency_100pct_epochs_fold0.csv')
        if flat_ep.exists() and not fold0_ep.exists():
            _shutil.copy2(str(flat_ep), str(fold0_ep))
            print(f"  ↳ Migrated flat 100pct epochs CSV → {fold0_ep}")

        print("✓ Legacy migration complete")

    # Build patient→label map once (lightweight JSON scan)
    print("\nBuilding patient fold splits...")
    data_root = config['data_root']
    patients_template = pathlib.Path(data_root) / "patients" / "patient_{p:03d}.json"
    patient_conditions = {}
    for pid in range(1, 470):
        p_path = pathlib.Path(str(patients_template).format(p=pid))
        if p_path.exists():
            try:
                with open(p_path, 'r') as f:
                    cond = json.load(f).get('condition', 'Unknown')
                patient_conditions[pid] = (0 if cond == 'Healthy' else 1 if 'Parkinson' in cond else 2)
            except:
                pass

    pids_sorted = sorted(patient_conditions)
    plabels = [patient_conditions[p] for p in pids_sorted]
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)
    fold_splits = list(skf.split(pids_sorted, plabels))

    all_fold_results = [None] * num_folds  # indexed by fold
    fractions = config.get('label_efficiency_fractions', [0.05, 0.10, 0.20, 0.50, 0.70, 1.0])
    expected_pcts = {int(round(f * 100)) for f in fractions}

    for fold_idx in range(num_folds):
        # ── Auto-detect completed folds from saved permutation JSON ────────
        perm_path = f'metrics/label_efficiency_permutation_fold{fold_idx}.json'
        if os.path.exists(perm_path):
            with open(perm_path, 'r') as f:
                perm_data = json.load(f)
            completed_pcts = set(perm_data.get('completed_fractions', []))
            prior_results = perm_data.get('completed_results', [])
            if completed_pcts >= expected_pcts and prior_results:
                all_fold_results[fold_idx] = prior_results
                print(f"\n{'='*70}")
                print(f"  SKIPPING Fold {fold_idx + 1} / {num_folds}  (all {len(expected_pcts)} fractions complete)")
                print(f"{'='*70}")
                continue

        print("\n" + "="*70)
        print(f"  K-FOLD CV  —  Fold {fold_idx + 1} / {num_folds}")
        print("="*70)

        # --- Phase 1: Pre-training (exclude this fold's test patients) --------
        existing_pretrain_ckpt = f'checkpoints/fold_{fold_idx}/best_contrastive_model.pth'
        if config.get('load_pretrained_checkpoint'):
            print("LOADING PRE-TRAINED ENCODER FROM CHECKPOINT (shared across folds)")
            pretrained_encoder, _ = load_pretrained_encoder(
                config['load_pretrained_checkpoint'], config, device
            )
        elif os.path.exists(existing_pretrain_ckpt):
            print(f"✓ Found existing pre-trained encoder for fold {fold_idx}, loading...")
            pretrained_encoder, _ = load_pretrained_encoder(
                existing_pretrain_ckpt, config, device
            )
        else:
            _, test_idx = fold_splits[fold_idx]
            exclude_ids = np.array([pids_sorted[i] for i in test_idx])
            print(f"Excluding {len(exclude_ids)} test-fold patients from pretraining")

            pretrained_encoder, _ = pretrain_self_supervised(
                config, exclude_patient_ids=exclude_ids, fold_idx=fold_idx
            )

        # --- Phase 2: Label Efficiency for this fold --------------------------
        fold_results = label_efficiency_experiment(pretrained_encoder, config, fold_idx=fold_idx)
        all_fold_results[fold_idx] = fold_results

    # --- Aggregate across ALL folds -------------------------------------------
    os.makedirs('metrics', exist_ok=True)

    # Filter to folds that have results (completed or just-processed)
    valid_fold_results = [r for r in all_fold_results if r is not None]
    n_complete = len(valid_fold_results)
    print(f"\n✓ {n_complete}/{num_folds} folds have results")

    summary_rows = []

    print("\n" + "="*70)
    print("K-FOLD CROSS-VALIDATION SUMMARY")
    print("="*70)
    print(f"{'%Labels':>8} | {'HC Acc':>10} | {'PD Acc':>10} | {'Combined':>10} | {'HC F1':>10} | {'PD F1':>10}")
    print("-"*70)

    for frac in fractions:
        pct = int(round(frac * 100))
        fold_vals = {k: [] for k in ['best_val_acc_hc', 'best_val_acc_pd', 'best_val_acc_combined', 'f1_hc', 'f1_pd']}
        for fold_res in valid_fold_results:
            for r in fold_res:
                if r['pct'] == pct:
                    for k in fold_vals:
                        fold_vals[k].append(r[k])
                    break

        if not fold_vals['best_val_acc_combined']:
            continue

        row = {'pct': pct}
        for k, vals in fold_vals.items():
            row[f'{k}_mean'] = float(np.mean(vals))
            row[f'{k}_std']  = float(np.std(vals))
        summary_rows.append(row)

        print(f"{pct:>7}% | "
              f"{row['best_val_acc_hc_mean']:.4f}±{row['best_val_acc_hc_std']:.4f} | "
              f"{row['best_val_acc_pd_mean']:.4f}±{row['best_val_acc_pd_std']:.4f} | "
              f"{row['best_val_acc_combined_mean']:.4f}±{row['best_val_acc_combined_std']:.4f} | "
              f"{row['f1_hc_mean']:.4f}±{row['f1_hc_std']:.4f} | "
              f"{row['f1_pd_mean']:.4f}±{row['f1_pd_std']:.4f}")

    # Save summary CSV
    summary_csv = 'metrics/label_efficiency_kfold_summary.csv'
    if summary_rows:
        with open(summary_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"\n\u2713 K-fold summary saved to {summary_csv}")
    else:
        print("\n⚠ No complete fold results to summarize yet")

    return all_fold_results



def inference_only(config, checkpoint_path=None):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if checkpoint_path is None:
        checkpoint_path = config.get('load_finetuned_checkpoint')
    
    if checkpoint_path is None:
        raise ValueError("No checkpoint path provided for inference. Set 'load_finetuned_checkpoint' in config or pass checkpoint_path.")
    
    print("="*60)
    print("INFERENCE MODE - Loading Fine-tuned Model")
    print("="*60)
    
    model, checkpoint = load_finetuned_classifier(checkpoint_path, config, device)
    model.eval()
    
    return model, checkpoint


if __name__ == "__main__":
    encoder, results = main()
