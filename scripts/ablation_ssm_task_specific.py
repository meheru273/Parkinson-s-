"""
Ablation Study: Task-Specific Models using Pure SSM (Mamba)
For Parkinson's Disease Detection

This script trains separate SSM models for each task to evaluate
how state-space models perform on individual motor tasks.
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
from typing import Dict, List, Tuple
import warnings
from scipy.signal import butter, filtfilt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import os
import csv
import matplotlib.pyplot as plt
from collections import defaultdict

warnings.filterwarnings("ignore", category=UserWarning)

# ============== Helper Functions ==============
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


# ============== Task-Specific Dataset ==============
class TaskSpecificDataset(Dataset):
    """Dataset for a specific task"""
    
    def __init__(self, data_root: str = None, task_name: str = None, 
                 window_size: int = 256, apply_downsampling=True,
                 apply_bandpass_filter=True,
                 left_samples=None, right_samples=None,
                 hc_vs_pd=None, pd_vs_dd=None, patient_ids=None):
        
        self.left_samples = []
        self.right_samples = []
        self.hc_vs_pd = []
        self.pd_vs_dd = []
        self.patient_ids = []
        self.task_name = task_name
        self.apply_downsampling = apply_downsampling
        self.apply_bandpass_filter = apply_bandpass_filter
        self.data_root = data_root
        self.window_size = window_size
        
        if data_root is not None and task_name is not None:
            self.patients_template = pathlib.Path(data_root) / "patients" / "patient_{p:03d}.json"
            self.timeseries_template = pathlib.Path(data_root) / "movement" / "timeseries" / "{N:03d}_{X}_{Y}.txt"
            self.patient_ids_list = list(range(1, 470))
            self._load_task_data()
        else:
            # Pre-loaded data
            if left_samples is not None:
                self.left_samples = np.array(left_samples) if not isinstance(left_samples, np.ndarray) else left_samples
            if right_samples is not None:
                self.right_samples = np.array(right_samples) if not isinstance(right_samples, np.ndarray) else right_samples
            if hc_vs_pd is not None:
                self.hc_vs_pd = np.array(hc_vs_pd) if not isinstance(hc_vs_pd, np.ndarray) else hc_vs_pd
            if pd_vs_dd is not None:
                self.pd_vs_dd = np.array(pd_vs_dd) if not isinstance(pd_vs_dd, np.ndarray) else pd_vs_dd
            if patient_ids is not None:
                self.patient_ids = np.array(patient_ids) if not isinstance(patient_ids, np.ndarray) else patient_ids
    
    def _load_task_data(self):
        """Load data for a specific task only"""
        print(f"Loading task: {self.task_name}")
        
        for patient_id in tqdm(self.patient_ids_list, desc=f"Loading {self.task_name}"):
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
                
                # Load only the specified task
                left_path = pathlib.Path(str(self.timeseries_template).format(
                    N=patient_id, X=self.task_name, Y="LeftWrist"))
                right_path = pathlib.Path(str(self.timeseries_template).format(
                    N=patient_id, X=self.task_name, Y="RightWrist"))
                
                if not (left_path.exists() and right_path.exists()):
                    continue
                
                left_data = np.loadtxt(left_path, delimiter=",")
                right_data = np.loadtxt(right_path, delimiter=",")
                
                # Preprocessing
                if left_data.shape[1] > 6:
                    left_data = left_data[:, :6]
                if left_data.shape[0] > 50:
                    left_data = left_data[50:, :]
                
                if right_data.shape[1] > 6:
                    right_data = right_data[:, :6]
                if right_data.shape[0] > 50:
                    right_data = right_data[50:, :]
                
                if self.apply_downsampling:
                    left_data = downsample(left_data)
                    right_data = downsample(right_data)
                
                if self.apply_bandpass_filter:
                    left_data = bandpass_filter(left_data)
                    right_data = bandpass_filter(right_data)
                
                # Create windows
                left_windows = create_windows(left_data, self.window_size, overlap=overlap)
                right_windows = create_windows(right_data, self.window_size, overlap=overlap)
                
                if left_windows is not None and right_windows is not None:
                    min_windows = min(len(left_windows), len(right_windows))
                    
                    for i in range(min_windows):
                        self.left_samples.append(left_windows[i])
                        self.right_samples.append(right_windows[i])
                        self.hc_vs_pd.append(hc_vs_pd_label)
                        self.pd_vs_dd.append(pd_vs_dd_label)
                        self.patient_ids.append(patient_id)
                        
            except Exception as e:
                continue
        
        self.left_samples = np.array(self.left_samples)
        self.right_samples = np.array(self.right_samples)
        self.hc_vs_pd = np.array(self.hc_vs_pd)
        self.pd_vs_dd = np.array(self.pd_vs_dd)
        self.patient_ids = np.array(self.patient_ids)
        
        print(f"Task {self.task_name}: {len(self.left_samples)} samples loaded")
    
    def get_kfold_splits(self, k=5):
        """Get K-fold splits at patient level"""
        patient_conditions = {}
        
        for idx, pid in enumerate(self.patient_ids):
            if pid not in patient_conditions:
                if self.hc_vs_pd[idx] == 0:
                    patient_conditions[pid] = 0  # HC
                elif self.hc_vs_pd[idx] == 1 and self.pd_vs_dd[idx] == 0:
                    patient_conditions[pid] = 1  # PD
                else:
                    patient_conditions[pid] = 2  # DD
        
        patient_list = list(patient_conditions.keys())
        patient_labels = [patient_conditions[p] for p in patient_list]
        
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
        fold_datasets = []
        
        for fold_id, (train_idx, test_idx) in enumerate(skf.split(patient_list, patient_labels)):
            train_patients = set([patient_list[i] for i in train_idx])
            test_patients = set([patient_list[i] for i in test_idx])
            
            train_mask = np.array([pid in train_patients for pid in self.patient_ids])
            test_mask = np.array([pid in test_patients for pid in self.patient_ids])
            
            train_dataset = TaskSpecificDataset(
                left_samples=self.left_samples[train_mask],
                right_samples=self.right_samples[train_mask],
                hc_vs_pd=self.hc_vs_pd[train_mask],
                pd_vs_dd=self.pd_vs_dd[train_mask],
                patient_ids=self.patient_ids[train_mask]
            )
            train_dataset.task_name = self.task_name
            
            test_dataset = TaskSpecificDataset(
                left_samples=self.left_samples[test_mask],
                right_samples=self.right_samples[test_mask],
                hc_vs_pd=self.hc_vs_pd[test_mask],
                pd_vs_dd=self.pd_vs_dd[test_mask],
                patient_ids=self.patient_ids[test_mask]
            )
            test_dataset.task_name = self.task_name
            
            fold_datasets.append((train_dataset, test_dataset))
        
        return fold_datasets
    
    def __len__(self):
        return len(self.left_samples) if hasattr(self, 'left_samples') and isinstance(self.left_samples, (list, np.ndarray)) else 0
    
    def __getitem__(self, idx):
        # Concatenate left and right wrist data (12 channels total)
        left_sample = torch.FloatTensor(self.left_samples[idx])
        right_sample = torch.FloatTensor(self.right_samples[idx])
        combined = torch.cat([left_sample, right_sample], dim=1)  # [timestep, 12]
        
        hc_vs_pd = torch.LongTensor([self.hc_vs_pd[idx]])
        pd_vs_dd = torch.LongTensor([self.pd_vs_dd[idx]])
        
        return combined, hc_vs_pd.squeeze(), pd_vs_dd.squeeze()


# ==============================================================================
# SSM Building Blocks (Mamba-style, from train_ssm.py)
# ==============================================================================

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


class AttentionPool(nn.Module):
    """
    Learnable attention pooling over the time dimension.
    Instead of averaging all timesteps equally, learns which timesteps
    are most diagnostically relevant (e.g. tremor bursts).
    """
    def __init__(self, model_dim: int):
        super().__init__()
        self.attention = nn.Linear(model_dim, 1)

    def forward(self, x):
        # x: (B, L, model_dim)
        weights = torch.softmax(self.attention(x), dim=1)  # (B, L, 1)
        return (weights * x).sum(dim=1)                    # (B, model_dim)


class MambaBlock(nn.Module):
    """
    Selective State-Space Model block (Mamba-style).

    Implements the core Mamba architecture from scratch in pure PyTorch:
      - Input projection with expansion
      - Causal depthwise 1D convolution
      - Selective scan with input-dependent Δ, B, C
      - Output gating and projection

    Args:
        model_dim:  Input/output dimension
        d_state:    SSM state dimension (N in the paper)
        expand:     Expansion factor for inner dimension
        d_conv:     Kernel size for the causal 1D convolution
        dropout:    Dropout rate
    """

    def __init__(
        self,
        model_dim: int,
        d_state: int = 16,
        expand: int = 2,
        d_conv: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.model_dim = model_dim
        self.d_state = d_state
        self.d_inner = model_dim * expand
        self.d_conv = d_conv

        # Input projection: project to 2 * d_inner (one for main path, one for gate)
        self.in_proj = nn.Linear(model_dim, self.d_inner * 2, bias=False)

        # Causal depthwise 1D conv
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,  # causal padding
            groups=self.d_inner,
            bias=True,
        )

        # SSM parameters — input-dependent projections
        # x → Δ (controls discretization step size, selective)
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + 1, bias=False)
        # Δ projection (rank-1 bottleneck → d_inner)
        self.dt_proj = nn.Linear(1, self.d_inner, bias=True)

        # Learnable A parameter (structured as log for stability)
        # Shape: (d_inner, d_state)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).expand(self.d_inner, -1)
        self.A_log = nn.Parameter(torch.log(A))

        # D parameter (skip connection scalar per channel)
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, model_dim, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: (B, L, model_dim)
        Returns:
            (B, L, model_dim)
        """
        B, L, _ = x.shape

        xz = self.in_proj(x)  # (B, L, 2 * d_inner)
        x_main, z = xz.chunk(2, dim=-1)  # each (B, L, d_inner)

        x_conv = x_main.transpose(1, 2)  # (B, d_inner, L)
        x_conv = self.conv1d(x_conv)[:, :, :L]  # causal: trim to original length
        x_conv = x_conv.transpose(1, 2)  # (B, L, d_inner)
        x_conv = F.silu(x_conv)

        # SSM parameter projections from convolved input
        x_ssm_params = self.x_proj(x_conv)  # (B, L, d_state*2 + 1)
        delta = x_ssm_params[:, :, :1]  # (B, L, 1)
        B_input = x_ssm_params[:, :, 1 : 1 + self.d_state]  # (B, L, d_state)
        C_input = x_ssm_params[:, :, 1 + self.d_state :]  # (B, L, d_state)

        # Discretize delta
        delta = self.dt_proj(delta)  # (B, L, d_inner)
        delta = F.softplus(delta)  # ensure positive

        # Get continuous A
        A = -torch.exp(self.A_log)
        # Selective scan (sequential for compatibility)
        y = self._selective_scan(x_conv, delta, A, B_input, C_input)

        # Skip connection with D
        y = y + self.D.unsqueeze(0).unsqueeze(0) * x_conv

        # Output gate (SiLU gate)
        y = y * F.silu(z)

        # Output projection
        y = self.out_proj(y)
        y = self.dropout(y)

        return y

    def _selective_scan(self, x, delta, A, B_input, C_input):
        """
        Sequential selective scan implementation with truncated BPTT.

        Args:
            x:        (B, L, d_inner) — input after conv
            delta:    (B, L, d_inner) — discretization step sizes
            A:        (d_inner, d_state) — state transition matrix (continuous)
            B_input:  (B, L, d_state) — input-dependent B
            C_input:  (B, L, d_state) — input-dependent C

        Returns:
            y: (B, L, d_inner)
        """
        B_batch, L, d_inner = x.shape
        chunk_size = 32  # gradient paths limited to 32 steps

        delta_A = torch.exp(delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))  # (B, L, d_inner, d_state)
        delta_B = delta.unsqueeze(-1) * B_input.unsqueeze(2)                    # (B, L, d_inner, d_state)

        h = torch.zeros(B_batch, d_inner, self.d_state, device=x.device, dtype=x.dtype)
        ys = []

        for start in range(0, L, chunk_size):
            end = min(start + chunk_size, L)
            h = h.detach()  # truncated BPTT: stop gradients between chunks
            for t in range(start, end):
                h = delta_A[:, t] * h + delta_B[:, t] * x[:, t].unsqueeze(-1)
                y_t = (h * C_input[:, t].unsqueeze(1)).sum(dim=-1)
                ys.append(y_t)

        return torch.stack(ys, dim=1)


class SSMLayer(nn.Module):
    """Single SSM layer: MambaBlock + residual + LayerNorm + FeedForward."""

    def __init__(
        self,
        model_dim: int,
        d_ff: int,
        d_state: int = 16,
        expand: int = 2,
        d_conv: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(model_dim)
        self.mamba = MambaBlock(model_dim, d_state, expand, d_conv, dropout)
        self.norm2 = nn.LayerNorm(model_dim)
        self.ffn = FeedForward(model_dim, d_ff, dropout)

    def forward(self, x):
        # Pre-norm Mamba block with residual
        x = x + self.mamba(self.norm1(x))
        # FFN (already has internal residual + layer norm)
        x = self.ffn(x)
        return x


class SSMStack(nn.Module):
    """Stack of N SSM layers."""

    def __init__(
        self,
        num_layers: int,
        model_dim: int,
        d_ff: int,
        d_state: int = 16,
        expand: int = 2,
        d_conv: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            SSMLayer(model_dim, d_ff, d_state, expand, d_conv, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


# ==============================================================================
# Pure SSM Model for Task-Specific Ablation
# ==============================================================================

class SSMModel(nn.Module):
    """
    Pure SSM (Mamba) model for task-specific classification.

    Takes combined 12-channel input (left+right wrist concatenated)
    matching the same interface as CNN1DModel and LSTMModel.

    Architecture:
        InputProjection(12→model_dim) → SSMStack → AttentionPool → ClassifierHeads
    """

    def __init__(
        self,
        input_channels: int = 12,
        model_dim: int = 128,
        num_layers: int = 4,
        d_ff: int = 512,
        dropout: float = 0.3,
        # SSM-specific
        ssm_d_state: int = 16,
        ssm_expand: int = 2,
        ssm_d_conv: int = 4,
    ):
        super().__init__()
        self.model_dim = model_dim

        # Input projection: 12 channels → model_dim
        self.input_projection = nn.Linear(input_channels, model_dim)
        self.input_norm = nn.LayerNorm(model_dim)
        self.input_dropout = nn.Dropout(dropout)

        # SSM stack
        self.ssm_stack = SSMStack(
            num_layers=num_layers,
            model_dim=model_dim,
            d_ff=d_ff,
            d_state=ssm_d_state,
            expand=ssm_expand,
            d_conv=ssm_d_conv,
            dropout=dropout,
        )

        # Attention pooling over timesteps
        self.pool = AttentionPool(model_dim)

        self.feature_dim = model_dim

        # Classification heads
        self.head_hc_vs_pd = nn.Sequential(
            nn.Linear(self.feature_dim, model_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim, 2),
        )

        self.head_pd_vs_dd = nn.Sequential(
            nn.Linear(self.feature_dim, model_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim, 2),
        )

    def get_features(self, x):
        """
        Args:
            x: (B, T, 12) — combined left+right wrist channels
        Returns:
            features: (B, model_dim)
        """
        # Project input channels to model dimension
        x = self.input_projection(x)       # (B, T, model_dim)
        x = self.input_norm(x)
        x = self.input_dropout(x)

        # Process through SSM stack
        x = self.ssm_stack(x)              # (B, T, model_dim)

        # Pool over timesteps
        features = self.pool(x)            # (B, model_dim)

        return features

    def forward(self, x):
        features = self.get_features(x)
        logits_hc_vs_pd = self.head_hc_vs_pd(features)
        logits_pd_vs_dd = self.head_pd_vs_dd(features)
        return logits_hc_vs_pd, logits_pd_vs_dd


# ============== Training Functions ==============
def calculate_metrics(y_true, y_pred, task_name="", verbose=True):
    if len(y_true) == 0:
        return {}
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    precision_avg, recall_avg, f1_avg, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )
    
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
        print(f"\n=== {task_name} ===")
        print(f"Accuracy: {accuracy:.4f}, Precision: {precision_avg:.4f}, "
              f"Recall: {recall_avg:.4f}, F1: {f1_avg:.4f}")
    
    return metrics


def train_epoch(model, dataloader, criterion_hc, criterion_pd, optimizer, device):
    model.train()
    train_loss = 0.0
    hc_pd_preds, hc_pd_labels = [], []
    pd_dd_preds, pd_dd_labels = [], []
    
    for batch in dataloader:
        data, hc_pd, pd_dd = batch
        data = data.to(device)
        hc_pd = hc_pd.to(device)
        pd_dd = pd_dd.to(device)
        
        optimizer.zero_grad()
        hc_pd_logits, pd_dd_logits = model(data)
        
        total_loss = 0
        loss_count = 0
        
        # HC vs PD loss
        valid_hc_pd_mask = (hc_pd != -1)
        if valid_hc_pd_mask.any():
            valid_logits = hc_pd_logits[valid_hc_pd_mask]
            valid_labels = hc_pd[valid_hc_pd_mask]
            loss = criterion_hc(valid_logits, valid_labels)
            total_loss += loss
            loss_count += 1
            
            preds = torch.argmax(valid_logits, dim=1)
            hc_pd_preds.extend(preds.cpu().numpy())
            hc_pd_labels.extend(valid_labels.cpu().numpy())
        
        # PD vs DD loss
        valid_pd_dd_mask = (pd_dd != -1)
        if valid_pd_dd_mask.any():
            valid_logits = pd_dd_logits[valid_pd_dd_mask]
            valid_labels = pd_dd[valid_pd_dd_mask]
            loss = criterion_pd(valid_logits, valid_labels)
            total_loss += loss
            loss_count += 1
            
            preds = torch.argmax(valid_logits, dim=1)
            pd_dd_preds.extend(preds.cpu().numpy())
            pd_dd_labels.extend(valid_labels.cpu().numpy())
        
        if loss_count > 0:
            avg_loss = total_loss / loss_count
            avg_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.25)
            optimizer.step()
            train_loss += avg_loss.item()
    
    train_loss /= len(dataloader)
    
    metrics_hc = calculate_metrics(hc_pd_labels, hc_pd_preds, verbose=False)
    metrics_pd = calculate_metrics(pd_dd_labels, pd_dd_preds, verbose=False)
    
    return train_loss, metrics_hc, metrics_pd


def validate_epoch(model, dataloader, criterion_hc, criterion_pd, device):
    model.eval()
    val_loss = 0.0
    hc_pd_preds, hc_pd_labels = [], []
    pd_dd_preds, pd_dd_labels = [], []
    
    with torch.no_grad():
        for batch in dataloader:
            data, hc_pd, pd_dd = batch
            data = data.to(device)
            hc_pd = hc_pd.to(device)
            pd_dd = pd_dd.to(device)
            
            hc_pd_logits, pd_dd_logits = model(data)
            
            total_loss = 0
            loss_count = 0
            
            # HC vs PD
            valid_hc_pd_mask = (hc_pd != -1)
            if valid_hc_pd_mask.any():
                valid_logits = hc_pd_logits[valid_hc_pd_mask]
                valid_labels = hc_pd[valid_hc_pd_mask]
                loss = criterion_hc(valid_logits, valid_labels)
                total_loss += loss
                loss_count += 1
                
                preds = torch.argmax(valid_logits, dim=1)
                hc_pd_preds.extend(preds.cpu().numpy())
                hc_pd_labels.extend(valid_labels.cpu().numpy())
            
            # PD vs DD
            valid_pd_dd_mask = (pd_dd != -1)
            if valid_pd_dd_mask.any():
                valid_logits = pd_dd_logits[valid_pd_dd_mask]
                valid_labels = pd_dd[valid_pd_dd_mask]
                loss = criterion_pd(valid_logits, valid_labels)
                total_loss += loss
                loss_count += 1
                
                preds = torch.argmax(valid_logits, dim=1)
                pd_dd_preds.extend(preds.cpu().numpy())
                pd_dd_labels.extend(valid_labels.cpu().numpy())
            
            if loss_count > 0:
                val_loss += (total_loss / loss_count).item()
    
    val_loss /= len(dataloader)
    
    metrics_hc = calculate_metrics(hc_pd_labels, hc_pd_preds, verbose=False)
    metrics_pd = calculate_metrics(pd_dd_labels, pd_dd_preds, verbose=False)
    
    return val_loss, metrics_hc, metrics_pd


# ============== SSM Task Training ==============
def train_task_model(config, task_name):
    """Train a Pure SSM model for a specific task"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f"Training SSM model for task: {task_name}")
    print(f"Device: {device}")
    print(f"{'='*60}")
    
    # Load task-specific dataset
    dataset = TaskSpecificDataset(
        data_root=config['data_root'],
        task_name=task_name,
        window_size=config['window_size'],
        apply_downsampling=config['apply_downsampling'],
        apply_bandpass_filter=config['apply_bandpass_filter']
    )
    
    if len(dataset) == 0:
        print(f"No data found for task {task_name}")
        return None
    
    # Get K-fold splits
    fold_datasets = dataset.get_kfold_splits(k=config['num_folds'])
    
    all_fold_results = []
    
    for fold_idx, (train_dataset, val_dataset) in enumerate(fold_datasets):
        print(f"\n--- Fold {fold_idx+1}/{config['num_folds']} ---")
        print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
        
        train_loader = DataLoader(
            train_dataset, batch_size=config['batch_size'],
            shuffle=True, num_workers=config['num_workers']
        )
        val_loader = DataLoader(
            val_dataset, batch_size=config['batch_size'],
            shuffle=False, num_workers=config['num_workers']
        )
        
        # Create SSM model
        model = SSMModel(
            input_channels=config['input_channels'],
            model_dim=config['model_dim'],
            num_layers=config['num_ssm_layers'],
            d_ff=config['d_ff'],
            dropout=config['dropout'],
            ssm_d_state=config['ssm_d_state'],
            ssm_expand=config['ssm_expand'],
            ssm_d_conv=config['ssm_d_conv'],
        ).to(device)
        
        # Print model size for first fold only
        if fold_idx == 0:
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"SSM Model — Total params: {total_params:,}, Trainable: {trainable_params:,}")
        
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config['num_epochs'], eta_min=1e-6
        )
        
        criterion_hc = nn.CrossEntropyLoss()
        criterion_pd = nn.CrossEntropyLoss()
        
        best_val_acc = 0.0
        best_metrics = None
        history = defaultdict(list)
        patience_counter = 0
        
        for epoch in range(config['num_epochs']):
            # Training
            train_loss, train_hc, train_pd = train_epoch(
                model, train_loader, criterion_hc, criterion_pd, optimizer, device
            )
            
            # Validation
            val_loss, val_hc, val_pd = validate_epoch(
                model, val_loader, criterion_hc, criterion_pd, device
            )
            
            scheduler.step()
            
            # Calculate combined accuracy
            val_acc_hc = val_hc.get('accuracy', 0)
            val_acc_pd = val_pd.get('accuracy', 0)
            val_acc_combined = (val_acc_hc + val_acc_pd) / 2
            
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['val_acc_hc'].append(val_acc_hc)
            history['val_acc_pd'].append(val_acc_pd)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, "
                      f"HC Acc={val_acc_hc:.4f}, PD Acc={val_acc_pd:.4f}")
            
            if val_acc_combined > best_val_acc:
                best_val_acc = val_acc_combined
                best_metrics = {
                    'hc_vs_pd': val_hc,
                    'pd_vs_dd': val_pd,
                    'combined_acc': val_acc_combined
                }
                patience_counter = 0
                
                # Save best model
                os.makedirs("models/ssm", exist_ok=True)
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'task': task_name,
                    'fold': fold_idx,
                    'best_acc': best_val_acc,
                    'config': config
                }, f"models/ssm/{task_name}_fold{fold_idx+1}.pth")
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= config.get('early_stop_patience', 15):
                print(f"Early stopping at epoch {epoch+1} (no improvement for "
                      f"{config.get('early_stop_patience', 15)} epochs)")
                break
        
        fold_result = {
            'fold': fold_idx + 1,
            'best_val_acc': best_val_acc,
            'best_metrics': best_metrics,
            'history': dict(history)
        }
        all_fold_results.append(fold_result)
        
        print(f"Fold {fold_idx+1} Best Combined Accuracy: {best_val_acc:.4f}")
    
    return all_fold_results


# ============== Ablation Orchestration ==============
def run_ablation_study(config):
    """Run ablation study for all tasks using SSM model"""
    
    tasks = config.get('tasks', [
        "CrossArms", "DrinkGlas", "Entrainment", "HoldWeight", "LiftHold",
        "PointFinger", "Relaxed", "StretchHold", "TouchIndex", "TouchNose"
    ])
    
    all_results = {}
    
    for task in tasks:
        print(f"\n{'#'*60}")
        print(f"ABLATION: SSM (Mamba) - {task}")
        print(f"{'#'*60}")
        
        results = train_task_model(config, task)
        
        if results is not None:
            all_results[task] = results
            
            # Calculate average metrics across folds
            avg_acc = np.mean([r['best_val_acc'] for r in results])
            std_acc = np.std([r['best_val_acc'] for r in results])
            print(f"\nSSM - {task}: Avg Acc = {avg_acc:.4f} ± {std_acc:.4f}")
    
    # Save summary
    save_ablation_summary(all_results, config)
    
    return all_results


def save_ablation_summary(results, config):
    """Save ablation study summary to CSV"""
    
    output_dir = config.get('output_dir', 'ablation_results_ssm')
    os.makedirs(output_dir, exist_ok=True)
    
    summary_file = os.path.join(output_dir, "ssm_task_specific_summary.csv")
    
    with open(summary_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Model', 'Task', 'Fold', 'HC_vs_PD_Acc', 'PD_vs_DD_Acc', 'Combined_Acc'])
        
        for task, fold_results in results.items():
            for fold_data in fold_results:
                hc_acc = fold_data['best_metrics']['hc_vs_pd'].get('accuracy', 0)
                pd_acc = fold_data['best_metrics']['pd_vs_dd'].get('accuracy', 0)
                combined = fold_data['best_val_acc']
                writer.writerow(['ssm', task, fold_data['fold'], 
                               f"{hc_acc:.4f}", f"{pd_acc:.4f}", f"{combined:.4f}"])
    
    print(f"\nSummary saved to: {summary_file}")
    
    # Also save aggregated results
    agg_file = os.path.join(output_dir, "ssm_task_specific_aggregated.csv")
    
    with open(agg_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Model', 'Task', 'Avg_Combined_Acc', 'Std_Combined_Acc', 
                        'Avg_HC_Acc', 'Avg_PD_Acc'])
        
        for task, fold_results in results.items():
            combined_accs = [r['best_val_acc'] for r in fold_results]
            hc_accs = [r['best_metrics']['hc_vs_pd'].get('accuracy', 0) for r in fold_results]
            pd_accs = [r['best_metrics']['pd_vs_dd'].get('accuracy', 0) for r in fold_results]
            
            writer.writerow([
                'ssm', task,
                f"{np.mean(combined_accs):.4f}", f"{np.std(combined_accs):.4f}",
                f"{np.mean(hc_accs):.4f}", f"{np.mean(pd_accs):.4f}"
            ])
    
    print(f"Aggregated results saved to: {agg_file}")


def plot_ablation_results(results, output_dir="ablation_results_ssm"):
    """Create visualization of SSM ablation study results"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare data for plotting
    tasks = list(results.keys())
    
    means = []
    stds = []
    hc_means = []
    pd_means = []
    
    for task in tasks:
        accs = [r['best_val_acc'] for r in results[task]]
        hc_accs = [r['best_metrics']['hc_vs_pd'].get('accuracy', 0) for r in results[task]]
        pd_accs = [r['best_metrics']['pd_vs_dd'].get('accuracy', 0) for r in results[task]]
        means.append(np.mean(accs))
        stds.append(np.std(accs))
        hc_means.append(np.mean(hc_accs))
        pd_means.append(np.mean(pd_accs))
    
    # Figure 1: Combined accuracy bar chart
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(tasks))
    bars = ax.bar(x, means, 0.6, yerr=stds, capsize=4, color='#2196F3', alpha=0.85,
                  edgecolor='#1565C0', linewidth=0.8)
    
    # Add value labels on bars
    for bar, mean_val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{mean_val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.set_ylabel('Combined Accuracy', fontsize=12)
    ax.set_xlabel('Task', fontsize=12)
    ax.set_title('Task-Specific SSM (Mamba) Model Performance', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(tasks, rotation=45, ha='right', fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 1.05)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "ssm_task_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Figure 2: HC vs PD and PD vs DD breakdown
    fig, ax = plt.subplots(figsize=(14, 6))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, hc_means, width, label='HC vs PD', color='#4CAF50', alpha=0.85)
    bars2 = ax.bar(x + width/2, pd_means, width, label='PD vs DD', color='#FF9800', alpha=0.85)
    
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_xlabel('Task', fontsize=12)
    ax.set_title('SSM (Mamba) Per-Task Breakdown: HC vs PD & PD vs DD', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(tasks, rotation=45, ha='right', fontsize=10)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 1.05)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "ssm_task_breakdown.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Plots saved to: {output_dir}/")


# ============== Main ==============
def main():
    """Main function for SSM ablation study"""
    
    config = {
        # Data settings
        'data_root': "/kaggle/input/parkinsons/pads-parkinsons-disease-smartwatch-dataset-1.0.0",
        'apply_downsampling': True,
        'apply_bandpass_filter': True,
        'window_size': 256,
        'input_channels': 12,  # 6 channels × 2 wrists
        
        # SSM model settings
        'model_dim': 128,
        'num_ssm_layers': 4,
        'd_ff': 512,
        'dropout': 0.3,
        'ssm_d_state': 16,
        'ssm_expand': 2,
        'ssm_d_conv': 4,
        
        # Training settings
        'num_folds': 5,
        'batch_size': 32,
        'learning_rate': 1e-3,
        'weight_decay': 1e-4,
        'num_epochs': 50,
        'num_workers': 0,
        'early_stop_patience': 15,
        
        # Ablation settings
        'tasks': ["CrossArms", "DrinkGlas", "Entrainment", "HoldWeight", "LiftHold",
                  "PointFinger", "Relaxed", "StretchHold", "TouchIndex", "TouchNose"],
        'output_dir': 'ablation_results_ssm',
    }
    
    print("=" * 60)
    print("ABLATION STUDY: Task-Specific Models (Pure SSM / Mamba)")
    print("=" * 60)
    print(f"\nSSM Config: model_dim={config['model_dim']}, layers={config['num_ssm_layers']}, "
          f"d_ff={config['d_ff']}, d_state={config['ssm_d_state']}, "
          f"expand={config['ssm_expand']}, d_conv={config['ssm_d_conv']}")
    
    results = run_ablation_study(config)
    
    # Generate plots
    plot_ablation_results(results, config['output_dir'])
    
    print("\n" + "=" * 60)
    print("SSM ABLATION STUDY COMPLETE")
    print("=" * 60)
    
    return results


if __name__ == "__main__":
    results = main()
