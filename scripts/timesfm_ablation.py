#!pip install "timesfm[torch] @ git+https://github.com/google-research/timesfm.git"
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
# K-FOLD SPLIT (from base model)
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
# DATASET
# ============================================================================

class ParkinsonsDataLoader(Dataset):
    """Dataset for Parkinson's disease detection with wrist sensor data."""
    
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
# LORA IMPLEMENTATION
# ============================================================================

class LoRALinear(nn.Module):
    """
    Low-Rank Adaptation (LoRA) for Linear layers.
    
    LoRA adds trainable low-rank matrices A and B to frozen pretrained weights.
    Output: h = W_0 * x + (B @ A) * x * (alpha / r)
    
    Paper: "LoRA: Low-Rank Adaptation of Large Language Models" (ICLR 2022)
    """
    
    def __init__(
        self,
        original_layer: nn.Linear,
        r: int = 8,
        alpha: float = 16,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.original_layer = original_layer
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        
        in_features = original_layer.in_features
        out_features = original_layer.out_features
        
        # Freeze original weights
        for param in self.original_layer.parameters():
            param.requires_grad = False
        
        # LoRA matrices
        self.lora_A = nn.Parameter(torch.zeros(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))
        
        # Initialize A with Kaiming uniform, B with zeros
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Original output (frozen)
        original_output = self.original_layer(x)
        
        # LoRA output
        lora_output = self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T
        
        return original_output + lora_output * self.scaling


class LoRAMultiHeadAttention(nn.Module):
    """
    LoRA wrapper for MultiHeadAttention.
    Applies LoRA to Q, K, V projections.
    """
    
    def __init__(
        self,
        original_mha: nn.MultiheadAttention,
        r: int = 8,
        alpha: float = 16,
        dropout: float = 0.1,
        apply_to_qkv: bool = True,
        apply_to_out: bool = True,
    ):
        super().__init__()
        
        self.original_mha = original_mha
        self.embed_dim = original_mha.embed_dim
        self.num_heads = original_mha.num_heads
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        
        # Freeze original MHA
        for param in self.original_mha.parameters():
            param.requires_grad = False
        
        # LoRA for in_proj (combined Q, K, V projection)
        if apply_to_qkv and hasattr(original_mha, 'in_proj_weight') and original_mha.in_proj_weight is not None:
            in_features = original_mha.in_proj_weight.shape[1]
            out_features = original_mha.in_proj_weight.shape[0]  # 3 * embed_dim
            
            self.lora_in_A = nn.Parameter(torch.zeros(r, in_features))
            self.lora_in_B = nn.Parameter(torch.zeros(out_features, r))
            
            nn.init.kaiming_uniform_(self.lora_in_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_in_B)
            self.has_in_lora = True
        else:
            self.has_in_lora = False
        
        # LoRA for out_proj
        if apply_to_out:
            out_proj = original_mha.out_proj
            self.lora_out_A = nn.Parameter(torch.zeros(r, out_proj.in_features))
            self.lora_out_B = nn.Parameter(torch.zeros(out_proj.out_features, r))
            
            nn.init.kaiming_uniform_(self.lora_out_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_out_B)
            self.has_out_lora = True
        else:
            self.has_out_lora = False
        
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
    
    def forward(self, query, key=None, value=None, **kwargs):
        if key is None:
            key = query
        if value is None:
            value = key
        
        # Get original output
        attn_output, attn_weights = self.original_mha(query, key, value, **kwargs)
        
        # Apply LoRA to input projection effect
        if self.has_in_lora:
            # Approximate the effect by adding LoRA adjustment
            batch_size, seq_len, _ = query.shape
            q_lora = self.lora_dropout(query) @ self.lora_in_A.T @ self.lora_in_B[:self.embed_dim].T
            attn_output = attn_output + q_lora * self.scaling
        
        # Apply LoRA to output projection
        if self.has_out_lora:
            out_lora = self.lora_dropout(attn_output) @ self.lora_out_A.T @ self.lora_out_B.T
            attn_output = attn_output + out_lora * self.scaling
        
        return attn_output, attn_weights


def apply_lora_to_model(model: nn.Module, config: dict) -> nn.Module:

    r = config.get('lora_r', 8)
    alpha = config.get('lora_alpha', 16)
    dropout = config.get('lora_dropout', 0.1)
    target_modules = config.get('target_modules', ['linear', 'projection', 'attn'])
    
    def should_apply_lora(name: str) -> bool:
        name_lower = name.lower()
        return any(target in name_lower for target in target_modules)
    
    # Apply LoRA to linear layers
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and should_apply_lora(name):
            # Get parent module and attribute name
            parts = name.split('.')
            parent = model
            for part in parts[:-1]:
                parent = getattr(parent, part)
            attr_name = parts[-1]
            
            # Replace with LoRA linear
            lora_linear = LoRALinear(module, r=r, alpha=alpha, dropout=dropout)
            setattr(parent, attr_name, lora_linear)
            print(f"  Applied LoRA to: {name}")
    
    return model


def get_lora_parameters(model: nn.Module) -> List[nn.Parameter]:
    """Get only the LoRA parameters from a model."""
    lora_params = []
    for name, param in model.named_parameters():
        if 'lora_' in name:
            lora_params.append(param)
    return lora_params


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    """Count model parameters."""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


# ============================================================================
# TIMESFM WRAPPER FOR CLASSIFICATION
# ============================================================================

class TimesFMFeatureExtractor(nn.Module):
    """
    Wrapper to extract embeddings from TimesFM for classification.
    
    TimesFM processes univariate time series, so we process each channel
    separately and then combine the embeddings.
    """
    
    def __init__(
        self,
        model_dim: int = 1280,  # TimesFM 2.5 hidden size
        patch_len: int = 32,    # TimesFM patch length
        freeze_backbone: bool = True,
        use_lora: bool = False,
        lora_config: dict = None,
    ):
        super().__init__()
        
        self.model_dim = model_dim
        self.patch_len = patch_len
        self.freeze_backbone = freeze_backbone
        self.use_lora = use_lora
        self._grad_check_done = False  # For one-time gradient diagnostic
        self.channel_projections = None  # Will become nn.ModuleList on first use
        self.emergency_projection = None  # Will become nn.Linear on first use
        
        # Try to load TimesFM
        self.timesfm_available = self._try_load_timesfm()
        
        if not self.timesfm_available:
            raise RuntimeError(
                "CRITICAL: TimesFM failed to load. "
                "Cannot proceed without the pretrained backbone — the experiment results would be invalid. "
                "Please check your network connection or 'timesfm' installation and try again."
            )
        else:
            if freeze_backbone and not use_lora:
                self._freeze_timesfm()
            elif use_lora:
                self._apply_lora(lora_config or {})
    
    def _try_load_timesfm(self) -> bool:
        """Attempt to load TimesFM model."""
        try:
            import timesfm
            
            # CORRECT: Provide the pretrained model path
            self.timesfm_model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
                "google/timesfm-2.5-200m-pytorch"
            )
            
            self.timesfm_module = self.timesfm_model
            # Register it so .to(device) propagates:
            if hasattr(self.timesfm_module, 'model') and isinstance(self.timesfm_module.model, nn.Module):
                self.register_module('_timesfm_inner', self.timesfm_module.model)
            print("✓ TimesFM loaded successfully!")
            print(f"  Model: TimesFM 2.5 (200M parameters)")
            
            return True
            
        except Exception as e:
            print(f"✗ TimesFM loading failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        
    def _freeze_timesfm(self):
        """Freeze TimesFM backbone."""
        try:
            # TimesFM 2.5 wraps the actual model in .model
            if hasattr(self.timesfm_module, 'model'):
                target_model = self.timesfm_module.model
            else:
                target_model = self.timesfm_module
                
            for param in target_model.parameters():
                param.requires_grad = False
            
            print("✓ TimesFM backbone frozen")
        except Exception as e:
            print(f"⚠️ Warning: Could not freeze TimesFM: {e}")
            print("   Continuing anyway (model may still work)")
    
    def _apply_lora(self, lora_config: dict):
        """Apply LoRA to TimesFM."""
        print("Applying LoRA to TimesFM...")
        
        # First freeze the backbone
        try:
            if hasattr(self.timesfm_module, 'model'):
                actual_model = self.timesfm_module.model
            else:
                actual_model = self.timesfm_module
            
            for param in actual_model.parameters():
                param.requires_grad = False
        except:
            print("⚠️ Could not freeze backbone, continuing...")
        
        # Apply LoRA to transformer layers
        r = lora_config.get('lora_r', 8)
        alpha = lora_config.get('lora_alpha', 16)
        dropout = lora_config.get('lora_dropout', 0.1)
        
        lora_count = 0
        
        try:
            # Get the actual model
            if hasattr(self.timesfm_module, 'model'):
                model_to_modify = self.timesfm_module.model
            else:
                model_to_modify = self.timesfm_module
            
            # Apply to transformer layers in stacked_xf
            if hasattr(model_to_modify, 'stacked_xf'):
                # This is the ModuleList of transformers
                for layer_idx, layer in enumerate(model_to_modify.stacked_xf):
                    # Each layer is a Transformer object
                    for name, module in layer.named_modules():
                        # Target attention projections and feed-forward layers
                        if isinstance(module, nn.Linear) and any(x in name for x in ['q_proj', 'k_proj', 'v_proj', 'out_proj', 'linear1', 'linear2']):
                             # Get parent
                             parts = name.split('.')
                             parent = layer
                             if len(parts) > 1:
                                 for part in parts[:-1]:
                                     parent = getattr(parent, part)
                             attr_name = parts[-1]
                             
                             # Replace with LoRA
                             lora_linear = LoRALinear(module, r=r, alpha=alpha, dropout=dropout)
                             setattr(parent, attr_name, lora_linear)
                             lora_count += 1
            else:
                 # Fallback generic search
                 for name, module in model_to_modify.named_modules():
                     if isinstance(module, nn.Linear) and any(x in name.lower() for x in ['attn', 'mlp', 'proj']):
                         # Get parent
                         parts = name.split('.')
                         parent = model_to_modify
                         for part in parts[:-1]:
                             parent = getattr(parent, part)
                         attr_name = parts[-1]
                         
                         # Replace with LoRA
                         lora_linear = LoRALinear(module, r=r, alpha=alpha, dropout=dropout)
                         setattr(parent, attr_name, lora_linear)
                         lora_count += 1
            
            if lora_count > 0:
                total_params = count_parameters(model_to_modify)
                trainable_params = count_parameters(model_to_modify, trainable_only=True)
                
                print(f"✓ Applied LoRA to {lora_count} layers")
                print(f"  Total params: {total_params:,}")
                print(f"  Trainable params: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
            else:
                print("⚠️ No layers found for LoRA application")
        
        except Exception as e:
            print(f"⚠️ LoRA application failed: {e}")
            import traceback
            traceback.print_exc()
            print("   Continuing with frozen backbone only")
    
    def extract_features(
        self, 
        left_wrist: torch.Tensor,  # (batch, seq_len, channels)
        right_wrist: torch.Tensor
    ) -> torch.Tensor:
        """
        Extract features from wrist sensor data using TimesFM.
        
        Args:
            left_wrist: Left wrist sensor data (batch, seq_len, 6)
            right_wrist: Right wrist sensor data (batch, seq_len, 6)
        
        Returns:
            Combined features (batch, feature_dim)
        """
        batch_size, seq_len, n_channels = left_wrist.shape
        device = left_wrist.device
        
        return self._extract_timesfm_features(left_wrist, right_wrist)
    
    def _extract_timesfm_features(self, left_wrist: torch.Tensor, right_wrist: torch.Tensor) -> torch.Tensor:
        """Extract features using TimesFM 2.5 with EFFICIENT BATCH PROCESSING.
        
        Processes all 12 channels (6 left + 6 right) in a single batched forward pass
        instead of 12 separate loops.
        """
        device = left_wrist.device
        batch_size, seq_len, _ = left_wrist.shape

        try:
            # Get the underlying model
            if hasattr(self.timesfm_module, 'model'):
                model = self.timesfm_module.model
            else:
                model = self.timesfm_module

            patch_len = 32
            
            combined_inputs = torch.cat([left_wrist, right_wrist], dim=-1)
            n_channels = 12
        
            flat_inputs = combined_inputs.permute(0, 2, 1).reshape(batch_size * n_channels, seq_len)
            
            # 3. Handle Padding (to multiple of patch_len)
            if seq_len % patch_len != 0:
                pad_len = patch_len - (seq_len % patch_len)
                padding = torch.zeros(batch_size * n_channels, pad_len, device=device)
                flat_inputs = torch.cat([flat_inputs, padding], dim=1)
            
            curr_len = flat_inputs.shape[1]
            num_patches = curr_len // patch_len
            patched_input = flat_inputs.view(batch_size * n_channels, num_patches, patch_len)
            
            # 5. Masking (0 = valid/observed data in TimesFM convention)
            patched_mask = torch.zeros_like(patched_input)
            
            # 6. Forward Pass (One single pass for everything!)
            if self.freeze_backbone:
                with torch.no_grad():
                    embeddings = self._run_backbone_forward_patched(model, patched_input, patched_mask)
            else:
                embeddings = self._run_backbone_forward_patched(model, patched_input, patched_mask)
            
            # embeddings: (B*12, Num_Patches, 1280)
            
            # 7. Pooling (Mean over patches)
            pooled = embeddings.mean(dim=1)  # (B*12, 1280)
            
            # 8. Detach if frozen
            if self.freeze_backbone:
                pooled = pooled.detach()
            
            # 9. Reshape back to separate channels
            # (B*12, 1280) -> (B, 12, 1280)
            pooled_reshaped = pooled.view(batch_size, n_channels, -1)
            
            # 10. Mean-pool across channels: (B, 12, 1280) -> (B, 1280)
            #     This reduces dimensionality from 15,360 to 1,280
            final_features = pooled_reshaped.mean(dim=1)

            return final_features

        except Exception as e:
            print(f"⚠️ TimesFM feature extraction failed: {e}")
            print("   Falling back to simple processing...")
            import traceback
            traceback.print_exc()

            # Emergency fallback: simple mean pooling
            left_flat = left_wrist.mean(dim=1)   # (batch, 6)
            right_flat = right_wrist.mean(dim=1)  # (batch, 6)
            combined = torch.cat([left_flat, right_flat], dim=-1)  # (batch, 12)

            # Lazy-init emergency projection (WITH gradients)
            # Output model_dim to match channel-pooled feature dimension
            if self.emergency_projection is None:
                self.emergency_projection = nn.Linear(12, self.model_dim).to(device)
                print(f"⚠️ Using emergency projection: 12 → {self.model_dim}")

            return self.emergency_projection(combined)

    def _run_backbone_forward_patched(self, model, patched_input, patched_mask) -> torch.Tensor:
        results, _ = model(patched_input, patched_mask)
        output_embeddings = results[1]
        return output_embeddings
    
    def set_freeze_state(self, frozen: bool):
        self.freeze_backbone = frozen

    def forward(self, left_wrist: torch.Tensor, right_wrist: torch.Tensor) -> torch.Tensor:
        return self.extract_features(left_wrist, right_wrist)


# ============================================================================
# CLASSIFICATION MODELS
# ============================================================================

class TimesFMClassifier(nn.Module):
    """
    Classification model using TimesFM as feature extractor.
    
    Supports two modes:
    1. Gradual Unfreezing: TimesFM starts frozen; top layers are progressively
       unfrozen after a warm-up period, with a lower LR than the classifier heads.
    2. LoRA Fine-tuning: TimesFM with LoRA adapters, train LoRA + MLP heads.
    """
    
    def __init__(
        self,
        model_dim: int = 1280,
        n_channels: int = 6,
        hidden_dim: int = 512,
        dropout: float = 0.1,
        freeze_backbone: bool = True,
        use_lora: bool = False,
        lora_config: dict = None,
    ):
        super().__init__()
        
        self.model_dim = model_dim
        self.n_channels = n_channels
        self.use_lora = use_lora
        
        # Feature extractor
        self.feature_extractor = TimesFMFeatureExtractor(
            model_dim=model_dim,
            freeze_backbone=freeze_backbone,
            use_lora=use_lora,
            lora_config=lora_config,
        )
        
        # Feature extractor now returns (B, model_dim) after channel mean-pooling
        # Project: model_dim -> hidden_dim -> hidden_dim//2 with aggressive dropout
        self.feature_projection = nn.Sequential(
            nn.Linear(model_dim, hidden_dim),            # 1280 -> 512
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, hidden_dim // 2),      # 512 -> 256
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.3),
        )
        
        # Classification heads take hidden_dim//2 = 256
        head_input_dim = hidden_dim // 2
        self.head_hc_vs_pd = nn.Sequential(
            nn.Linear(head_input_dim, head_input_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(head_input_dim // 2, 2)
        )
        
        self.head_pd_vs_dd = nn.Sequential(
            nn.Linear(head_input_dim, head_input_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(head_input_dim // 2, 2)
        )
    
    def get_features(self, left_wrist: torch.Tensor, right_wrist: torch.Tensor) -> torch.Tensor:
        """Extract features for visualization."""
        raw_features = self.feature_extractor(left_wrist, right_wrist)
        projected_features = self.feature_projection(raw_features)
        return projected_features
    
    def forward(
        self, 
        left_wrist: torch.Tensor, 
        right_wrist: torch.Tensor,
        device=None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            left_wrist: Left wrist sensor data (batch, seq_len, 6)
            right_wrist: Right wrist sensor data (batch, seq_len, 6)
        
        Returns:
            logits_hc_vs_pd: Logits for HC vs PD classification
            logits_pd_vs_dd: Logits for PD vs DD classification
        """
        features = self.get_features(left_wrist, right_wrist)
        
        logits_hc_vs_pd = self.head_hc_vs_pd(features)
        logits_pd_vs_dd = self.head_pd_vs_dd(features)
        
        return logits_hc_vs_pd, logits_pd_vs_dd



# ============================================================================
# EVALUATION FUNCTIONS
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
                if task_name == "PD vs DD":
                    label_name = "PD" if label == 0 else ("DD" if label == 1 else f"Class_{label}")
                print(f"{label_name}: Precision={precision[i]:.4f}, Recall={recall[i]:.4f}, F1={f1[i]:.4f}, Support={support[i]}")
        
        print("Confusion Matrix:")
        print(cm)
    
    return metrics


def save_fold_metric(fold_idx, fold_suffix, best_epoch, best_val_acc,
                     fold_metrics_hc, fold_metrics_pd, output_dir="metrics"):
    """Save metrics to CSV files."""
    os.makedirs(output_dir, exist_ok=True)

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
        hc_filename = f"{output_dir}/hc_vs_pd_metrics{fold_suffix}.csv"
        write_csv(hc_filename, fold_metrics_hc)
        print(f"✓ HC vs PD metrics saved: {hc_filename}")

    if fold_metrics_pd:
        pd_filename = f"{output_dir}/pd_vs_dd_metrics{fold_suffix}.csv"
        write_csv(pd_filename, fold_metrics_pd)
        print(f"✓ PD vs DD metrics saved: {pd_filename}")


def save_epoch_metrics(epoch, fold_idx, fold_metrics_hc, fold_metrics_pd, output_dir="metrics"):
    """Save metrics after each epoch to prevent data loss - FIXED VERSION."""
    os.makedirs(output_dir, exist_ok=True)
    fold_suffix = f"_fold_{fold_idx+1}"
    
    # Save immediately after each epoch
    save_fold_metric(fold_idx, fold_suffix, epoch, 0, 
                    fold_metrics_hc, fold_metrics_pd, output_dir)


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


def plot_roc_curves(labels, predictions, probabilities, output_path):
    plt.figure(figsize=(10, 8))
    
    fpr, tpr, _ = roc_curve(labels, probabilities)
    roc_auc = auc(fpr, tpr)
    
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
                        c='blue', label=f'HC (n={np.sum(hc_mask)})', alpha=0.6, s=50)
        if np.any(pd_mask):
            plt.scatter(features_hc_pd[pd_mask,0], features_hc_pd[pd_mask,1], 
                        c='red', label=f'PD (n={np.sum(pd_mask)})', alpha=0.6, s=50)

        plt.title("t-SNE: HC vs PD")
        plt.xlabel("t-SNE Component 1")
        plt.ylabel("t-SNE Component 2")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir,"tsne_hc_vs_pd.png"), dpi=150, bbox_inches='tight')
        plt.close()

    if np.any(valid_pd_dd):
        plt.figure(figsize=(8, 6))
        features_pd_dd = features_2d[valid_pd_dd]
        labels_pd_dd = pd_dd_labels[valid_pd_dd]

        pd_mask = labels_pd_dd == 0
        dd_mask = labels_pd_dd == 1
        
        if np.any(pd_mask):
            plt.scatter(features_pd_dd[pd_mask,0], features_pd_dd[pd_mask,1], 
                        c='green', label=f'PD (n={np.sum(pd_mask)})', alpha=0.6, s=50)
        if np.any(dd_mask):
            plt.scatter(features_pd_dd[dd_mask,0], features_pd_dd[dd_mask,1], 
                        c='orange', label=f'DD (n={np.sum(dd_mask)})', alpha=0.6, s=50)

        plt.title("t-SNE: PD vs DD")
        plt.xlabel("t-SNE Component 1")
        plt.ylabel("t-SNE Component 2")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir,"tsne_pd_vs_dd.png"), dpi=150, bbox_inches='tight')
        plt.close()

    return features_2d


# ============================================================================
# CHECKPOINT LOADING AND SAVING
# ============================================================================

def load_checkpoint(checkpoint_path, model, optimizer=None, device='cuda'):
    """
    Load checkpoint and return the state.
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: Model to load state into
        optimizer: Optimizer to load state into (optional)
        device: Device to load model to
        
    Returns:
        Dictionary with checkpoint information
    """
    print(f"\n{'='*60}")
    print(f"Loading checkpoint from: {checkpoint_path}")
    print(f"{'='*60}")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Load checkpoint (weights_only=False needed for checkpoints with numpy objects)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'],strict=False)
    print("✓ Model state loaded")
    
    # Load optimizer state if provided
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("✓ Optimizer state loaded")
    
    # Extract checkpoint info
    start_epoch = checkpoint.get('epoch', 0) + 1  # Start from next epoch
    fold = checkpoint.get('fold', 1)
    val_acc_combined = checkpoint.get('val_acc_combined', 0)
    val_acc_hc = checkpoint.get('val_acc_hc', 0)
    val_acc_pd = checkpoint.get('val_acc_pd', 0)
    
    print(f"\nCheckpoint Info:")
    print(f"  Fold: {fold}")
    print(f"  Last completed epoch: {start_epoch - 1}")
    print(f"  Will resume from epoch: {start_epoch}")
    print(f"  Best val accuracy - Combined: {val_acc_combined:.4f}")
    print(f"  Best val accuracy - HC vs PD: {val_acc_hc:.4f}")
    print(f"  Best val accuracy - PD vs DD: {val_acc_pd:.4f}")
    
    return {
        'start_epoch': start_epoch,
        'fold': fold,
        'best_val_acc': val_acc_combined,
        'config': checkpoint.get('config', {}),
        'ablation_type': checkpoint.get('ablation_type', 'mlp_finetune')
    }


def save_checkpoint(model, optimizer, fold, epoch, val_acc_combined, val_acc_hc, val_acc_pd,
                   config, ablation_type, save_path):
    """Save checkpoint with all necessary information."""
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'fold': fold,
        'epoch': epoch,
        'val_acc_combined': val_acc_combined,
        'val_acc_hc': val_acc_hc,
        'val_acc_pd': val_acc_pd,
        'config': config,
        'ablation_type': ablation_type,
    }, save_path)


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def train_single_epoch(model, dataloader, criterion_hc, criterion_pd, optimizer, device, scheduler=None):
    """Train for one epoch."""
    model.train()
    train_loss = 0.0
    hc_pd_train_pred, hc_pd_train_labels = [], []
    pd_dd_train_pred, pd_dd_train_labels = [], []
    grad_checked = False
    
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
        
        if loss_count > 0:
            avg_loss = total_loss / loss_count
            avg_loss.backward()
            
            # One-time gradient flow diagnostic
            if not grad_checked:
                grad_checked = True
                has_grads = False
                for name, param in model.named_parameters():
                    if param.requires_grad and param.grad is not None:
                        grad_norm = param.grad.norm().item()
                        if grad_norm > 0:
                            has_grads = True
                            break
                if has_grads:
                    print("\n✓ Gradient check PASSED: MLP head gradients are flowing")
                else:
                    print("\n✗ Gradient check FAILED: No gradients detected! Training will collapse.")
                    print("  Check that feature extractor output is not wrapped in torch.no_grad()")
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Step scheduler per batch if it's OneCycleLR
            if scheduler is not None and not isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step()
                
            train_loss += avg_loss.item()
    
    train_loss /= len(dataloader)
    
    train_metrics_hc = calculate_metrics(hc_pd_train_labels, hc_pd_train_pred, 
                                        "Training HC vs PD", verbose=False)
    train_metrics_pd = calculate_metrics(pd_dd_train_labels, pd_dd_train_pred, 
                                        "Training PD vs DD", verbose=False)
    
    return train_loss, train_metrics_hc, train_metrics_pd


def validate_single_epoch(model, dataloader, criterion_hc, criterion_pd, device):
    """Validate for one epoch."""
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
    """Extract features for visualization."""
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


# ============================================================================
# GRADUAL UNFREEZING HELPER
# ============================================================================

def unfreeze_top_layers(model: nn.Module, n_layers: int) -> int:
    """
    Unfreeze only the last `n_layers` of TimesFM's stacked_xf transformer stack.

    Everything else (embedding layers, lower transformer blocks, etc.) stays frozen,
    so the backbone still acts as a strong regulariser while the top layers can
    adapt to the new task.

    Args:
        model: The TimesFMClassifier instance.
        n_layers: How many layers from the top of stacked_xf to unfreeze.

    Returns:
        Number of parameters unfrozen.
    """
    try:
        stacked_xf = model.feature_extractor.timesfm_module.model.stacked_xf
    except AttributeError:
        print("⚠️  unfreeze_top_layers: could not find stacked_xf — skipping.")
        return 0

    total_layers = len(stacked_xf)
    unfreeze_from = max(0, total_layers - n_layers)

    newly_unfrozen = 0
    for idx, layer in enumerate(stacked_xf):
        if idx >= unfreeze_from:
            for param in layer.parameters():
                if not param.requires_grad:
                    param.requires_grad = True
                    newly_unfrozen += param.numel()

    print(f"✓ Unfrozen top {n_layers}/{total_layers} transformer layers "
          f"({newly_unfrozen:,} additional parameters now trainable)")
    return newly_unfrozen


def _build_optimizer_with_discriminative_lr(model: nn.Module, config: dict) -> optim.Optimizer:
    """
    Build an AdamW optimizer with separate param groups for backbone vs heads.
    Used both at initial setup (gradual_unfreeze with all frozen) and when
    unfreezing kicks in mid-training.
    """
    backbone_lr = config.get('backbone_lr', 2e-5)
    head_lr = config.get('head_lr', 1e-3)
    weight_decay = config.get('weight_decay', 1e-4)

    backbone_params, head_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'feature_extractor.' in name:
            backbone_params.append(param)
        else:
            head_params.append(param)

    param_groups = []
    if backbone_params:
        param_groups.append({'params': backbone_params, 'lr': backbone_lr})
    if head_params:
        param_groups.append({'params': head_params, 'lr': head_lr})

    if not param_groups:
        # Defensive: should never happen, but avoid crashing
        param_groups = [{'params': list(model.parameters()), 'lr': head_lr}]

    return optim.AdamW(param_groups, weight_decay=weight_decay)


# ============================================================================
# MAIN TRAINING PIPELINE WITH CHECKPOINT RESUMPTION
# ============================================================================

def train_timesfm_model(config, resume_from_checkpoint=None):
    """
    Train TimesFM-based model for Parkinson's disease detection.
    NOW SUPPORTS CHECKPOINT RESUMPTION AND SAVES METRICS AFTER EACH EPOCH.
    
    Args:
        config: Configuration dictionary
        resume_from_checkpoint: Path to checkpoint to resume from (optional)
    
    Returns:
        List of fold results
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    ablation_type = config.get('ablation_type', 'mlp_finetune')
    print(f"\n{'='*60}")
    print(f"ABLATION STUDY: TimesFM with {ablation_type.upper()}")
    print(f"{'='*60}")
    
    # Create output directories
    output_base = config.get('output_dir', f'results/timesfm_{ablation_type}')
    os.makedirs(output_base, exist_ok=True)
    os.makedirs(f"{output_base}/metrics", exist_ok=True)
    os.makedirs(f"{output_base}/checkpoints", exist_ok=True)
    os.makedirs(f"{output_base}/plots", exist_ok=True)
    
    # Load dataset
    full_dataset = ParkinsonsDataLoader(
        config['data_root'],
        apply_dowsampling=config.get('apply_downsampling', True),
        apply_bandpass_filter=config.get('apply_bandpass_filter', True)
    )
    
    fold_datasets = full_dataset.get_train_test_split(split_type=3, k=config.get('num_folds', 5))
    num_folds = len(fold_datasets)
    max_folds = config.get('max_folds_to_train', num_folds)
    
    all_fold_results = []
    
    # Determine which fold to process
    start_fold = 0
    checkpoint_info = None
    
    if resume_from_checkpoint:
        print(f"\n{'='*60}")
        print(f"Loading checkpoint metadata from: {resume_from_checkpoint}")
        print(f"{'='*60}")
        
        if not os.path.exists(resume_from_checkpoint):
            raise FileNotFoundError(f"Checkpoint not found: {resume_from_checkpoint}")
        
        checkpoint_data = torch.load(resume_from_checkpoint, map_location=device, weights_only=False)
        
        start_fold = checkpoint_data.get('fold', 1) - 1  # Convert to 0-indexed
        _ckpt_start_epoch = checkpoint_data.get('epoch', 0) + 1
        _ckpt_best_val_acc = checkpoint_data.get('val_acc_combined', 0)
        _ckpt_best_hc = checkpoint_data.get('val_acc_hc', 0)
        _ckpt_best_pd = checkpoint_data.get('val_acc_pd', 0)
        
        print(f"\nCheckpoint Info:")
        print(f"  Fold: {start_fold + 1}")
        print(f"  Last completed epoch: {_ckpt_start_epoch - 1}")
        print(f"  Will resume from epoch: {_ckpt_start_epoch}")
        print(f"  Best val accuracy - Combined: {_ckpt_best_val_acc:.4f}")
        print(f"  Best val accuracy - HC vs PD: {_ckpt_best_hc:.4f}")
        print(f"  Best val accuracy - PD vs DD: {_ckpt_best_pd:.4f}")
        
        # Update config from checkpoint if needed
        saved_config = checkpoint_data.get('config', {})
        if saved_config:
            print(f"\nUsing configuration from checkpoint:")
            for key in ['model_dim', 'hidden_dim', 'dropout', 'batch_size', 'learning_rate']:
                if key in saved_config:
                    config[key] = saved_config[key]
                    print(f"  {key}: {config[key]}")
        
        # Store checkpoint data for later loading into the fold model
        checkpoint_info = {
            'start_epoch': _ckpt_start_epoch,
            'fold': start_fold + 1,
            'best_val_acc': _ckpt_best_val_acc,
            'config': saved_config,
            'model_state_dict': checkpoint_data['model_state_dict'],
            'optimizer_state_dict': checkpoint_data.get('optimizer_state_dict', None),
        }
    
    for fold_idx in range(start_fold, min(num_folds, max_folds)):
        print(f"\n{'='*60}")
        print(f"Starting Fold {fold_idx+1}/{num_folds}")
        print(f"{'='*60}")
        
        train_dataset, val_dataset = fold_datasets[fold_idx]
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=config.get('batch_size', 32), 
            shuffle=True, 
            num_workers=config.get('num_workers', 0)
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=config.get('batch_size', 32), 
            shuffle=False, 
            num_workers=config.get('num_workers', 0)
        )
        
        if ablation_type == 'gradual_unfreeze':
            print("\nMode: Gradual Unfreezing (backbone starts frozen; top layers unfreeze after warm-up)")
            model = TimesFMClassifier(
                model_dim=config.get('model_dim', 1280),
                hidden_dim=config.get('hidden_dim', 512),
                dropout=config.get('dropout', 0.1),
                freeze_backbone=True,   # Start fully frozen
                use_lora=False,
            ).to(device)
            
        elif ablation_type == 'lora':
            print("\nMode: LoRA Fine-tuning")
            lora_config = {
                'lora_r': config.get('lora_r', 8),
                'lora_alpha': config.get('lora_alpha', 16),
                'lora_dropout': config.get('lora_dropout', 0.1),
            }
            model = TimesFMClassifier(
                model_dim=config.get('model_dim', 1280),
                hidden_dim=config.get('hidden_dim', 512),
                dropout=config.get('dropout', 0.1),
                freeze_backbone=False,
                use_lora=True,
                lora_config=lora_config,
            ).to(device)
            
        else:
            raise ValueError(f"Unknown ablation_type: {ablation_type}. Valid options: 'gradual_unfreeze', 'lora'")
        
        # Print parameter counts
        total_params = count_parameters(model)
        trainable_params = count_parameters(model, trainable_only=True)
        print(f"\nModel Parameters:")
        print(f"  Total: {total_params:,}")
        print(f"  Trainable: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
        
        print("  Running warmup forward pass to initialize all layers...")
        with torch.no_grad():
            dummy_left = torch.randn(1, 256, 6).to(device)
            dummy_right = torch.randn(1, 256, 6).to(device)
            _ = model(dummy_left, dummy_right, device)
        
        # Re-count after lazy init
        total_params = count_parameters(model)
        trainable_params = count_parameters(model, trainable_only=True)
        print(f"  After init - Total: {total_params:,}, Trainable: {trainable_params:,}")
        
        # Optimizer — discriminative LRs are always used for gradual_unfreeze.
        # At epoch 0 the backbone is fully frozen so backbone_params will be empty
        # (no grad), and only the head params actually enter the optimiser.
        # When unfreezing kicks in we rebuild the optimiser (see epoch loop).
        if ablation_type == 'gradual_unfreeze':
            print(f"  Using DISCRIMINATIVE Learning Rates: Backbone={config.get('backbone_lr', 2e-5)}, "
                  f"Heads={config.get('head_lr', 1e-3)}")
            optimizer = _build_optimizer_with_discriminative_lr(model, config)
            
        else:
            # Standard single learning rate
            trainable_params_list = [p for p in model.parameters() if p.requires_grad]
            optimizer = optim.AdamW(
                trainable_params_list, 
                lr=config.get('learning_rate', 1e-4),
                weight_decay=config.get('weight_decay', 1e-4)
            )
        
        # Scheduler - Use OneCycleLR if warmup is requested (better for fine-tuning)
        num_epochs = config.get('num_epochs', 50)
        warmup_epochs = config.get('warmup_epochs', 0)
        
        if warmup_epochs > 0:
            print(f"  Using OneCycleLR scheduler with {warmup_epochs} warmup epochs")
            steps_per_epoch = len(train_loader)
            scheduler = optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=[g['lr'] for g in optimizer.param_groups],
                steps_per_epoch=steps_per_epoch,
                epochs=num_epochs,
                pct_start=warmup_epochs/num_epochs,
                div_factor=25.0,
                final_div_factor=1000.0
            )
        else:
            print("  Using ReduceLROnPlateau scheduler")
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=5
            )
        
        # Load checkpoint if resuming this specific fold
        start_epoch = 0
        best_val_acc = 0.0
        
        if resume_from_checkpoint and fold_idx == start_fold and checkpoint_info is not None:
            # Load model/optimizer state from pre-loaded checkpoint data
            print(f"\nLoading checkpoint weights into model...")
            model.load_state_dict(checkpoint_info['model_state_dict'])
            print("✓ Model state loaded")
            
            if checkpoint_info.get('optimizer_state_dict') is not None:
                try:
                    optimizer.load_state_dict(checkpoint_info['optimizer_state_dict'])
                    print("✓ Optimizer state loaded")
                except Exception as e:
                    print(f"⚠️ Could not load optimizer state: {e}")
                    print("  Continuing with fresh optimizer")
            
            start_epoch = checkpoint_info['start_epoch']
            best_val_acc = checkpoint_info['best_val_acc']
            print(f"\n✓ Resuming training from epoch {start_epoch}")
            
            # Free checkpoint data to save memory
            del checkpoint_info['model_state_dict']
            if 'optimizer_state_dict' in checkpoint_info:
                del checkpoint_info['optimizer_state_dict']
        
        label_smoothing = config.get('label_smoothing', 0.1)
        criterion_hc = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        criterion_pd = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        
        history = defaultdict(list)
        best_epoch = start_epoch
        patience_counter = 0
        early_stop_patience = config.get('early_stop_patience', 10)
        
        fold_metrics_hc = []
        fold_metrics_pd = []
        
        best_hc_pd_probs = None
        best_hc_pd_preds = None
        best_hc_pd_labels = None
        best_pd_dd_probs = None
        best_pd_dd_preds = None
        best_pd_dd_labels = None
        
        for epoch in range(start_epoch, num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")

            # ── Gradual unfreezing ──────────────────────────────────────────
            if ablation_type == 'gradual_unfreeze':
                unfreeze_after = config.get('unfreeze_after_epoch', 10)
                n_unfreeze = config.get('unfreeze_n_layers', 2)
                if epoch == unfreeze_after:
                    print(f"\n▶ Epoch {epoch + 1}: Triggering gradual unfreeze "
                          f"(top {n_unfreeze} layers of stacked_xf)")
                    unfreeze_top_layers(model, n_unfreeze)
                    # CRITICAL: Update freeze flag so forward pass enables gradients
                    model.feature_extractor.set_freeze_state(False)
                    # Rebuild optimiser so unfrozen params are included
                    optimizer = _build_optimizer_with_discriminative_lr(model, config)
                    # Rebuild scheduler for the remaining epochs
                    remaining_epochs = num_epochs - epoch
                    warmup_epochs = config.get('warmup_epochs', 0)
                    if warmup_epochs > 0:
                        steps_per_epoch = len(train_loader)
                        scheduler = optim.lr_scheduler.OneCycleLR(
                            optimizer,
                            max_lr=[g['lr'] for g in optimizer.param_groups],
                            steps_per_epoch=steps_per_epoch,
                            epochs=remaining_epochs,
                            pct_start=min(2 / remaining_epochs, 0.3),
                            div_factor=10.0,
                            final_div_factor=100.0,
                        )
                    else:
                        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                            optimizer, mode='min', factor=0.5, patience=5
                        )
                    # Log updated parameter counts
                    total_params = count_parameters(model)
                    trainable_params = count_parameters(model, trainable_only=True)
                    print(f"  Updated — Total: {total_params:,}, "
                          f"Trainable: {trainable_params:,} "
                          f"({100*trainable_params/total_params:.2f}%)")
            # ───────────────────────────────────────────────────────────────
            
            # Training
            train_loss, train_metrics_hc, train_metrics_pd = train_single_epoch(
                model, train_loader, criterion_hc, criterion_pd, optimizer, device, scheduler
            )
            
            # Validation
            val_results = validate_single_epoch(
                model, val_loader, criterion_hc, criterion_pd, device
            )
            val_loss, hc_pd_val_pred, hc_pd_val_labels, hc_pd_val_probs, \
            pd_dd_val_pred, pd_dd_val_labels, pd_dd_val_probs = val_results
            
            # Calculate metrics
            print("\n" + "="*60)
            val_metrics_hc = calculate_metrics(
                hc_pd_val_labels, hc_pd_val_pred,
                f"Fold {fold_idx+1} Validation HC vs PD",
                verbose=True
            )
            val_metrics_pd = calculate_metrics(
                pd_dd_val_labels, pd_dd_val_pred,
                f"Fold {fold_idx+1} Validation PD vs DD",
                verbose=True
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
            
            # *** CRITICAL FIX: Save metrics after EACH epoch ***
            if config.get('save_metrics', True):
                save_epoch_metrics(
                    epoch + 1, fold_idx, fold_metrics_hc, fold_metrics_pd,
                    output_dir=f"{output_base}/metrics"
                )
            
            val_acc_hc = val_metrics_hc.get('accuracy', 0)
            val_acc_pd = val_metrics_pd.get('accuracy', 0)
            val_acc_combined = (val_acc_hc + val_acc_pd) / 2
            
            train_acc_hc = train_metrics_hc.get('accuracy', 0)
            train_acc_pd = train_metrics_pd.get('accuracy', 0)
            
            # Step scheduler if it's Plateau (OneCycleLR is stepped in train_single_epoch)
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            
            # Save history
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
            
            # Save best model + early stopping
            if val_acc_combined > best_val_acc:
                best_val_acc = val_acc_combined
                best_epoch = epoch + 1
                patience_counter = 0
                
                if hc_pd_val_probs:
                    best_hc_pd_probs = np.array(hc_pd_val_probs)
                    best_hc_pd_preds = np.array(hc_pd_val_pred)
                    best_hc_pd_labels = np.array(hc_pd_val_labels)
                
                if pd_dd_val_probs:
                    best_pd_dd_probs = np.array(pd_dd_val_probs)
                    best_pd_dd_preds = np.array(pd_dd_val_pred)
                    best_pd_dd_labels = np.array(pd_dd_val_labels)
                
                model_save_path = f'{output_base}/checkpoints/best_model_fold_{fold_idx+1}.pth'
                save_checkpoint(
                    model, optimizer, fold_idx + 1, epoch, val_acc_combined,
                    val_acc_hc, val_acc_pd, config, ablation_type, model_save_path
                )
                print(f"✓ New best model saved: {model_save_path}")
            else:
                patience_counter += 1
                print(f"  No improvement ({patience_counter}/{early_stop_patience})")
            
            # Early stopping check
            if patience_counter >= early_stop_patience:
                print(f"\n⏹ Early stopping at epoch {epoch+1}: "
                      f"no improvement for {early_stop_patience} epochs. "
                      f"Best epoch was {best_epoch} with val_acc={best_val_acc:.4f}")
                break
        
        # Extract features for visualization
        fold_features, fold_hc_pd_labels, fold_pd_dd_labels = extract_features(
            model, val_loader, device
        )
        
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
        
        # Create plots
        if config.get('create_plots', True):
            plot_dir = f"{output_base}/plots/fold_{fold_idx+1}"
            os.makedirs(plot_dir, exist_ok=True)
            
            plot_loss(history['train_loss'], history['val_loss'], f"{plot_dir}/loss.png")
            
            if best_hc_pd_probs is not None and len(best_hc_pd_labels) > 0:
                plot_roc_curves(
                    best_hc_pd_labels, best_hc_pd_preds, best_hc_pd_probs,
                    f"{plot_dir}/roc_hc_vs_pd.png"
                )
            
            if best_pd_dd_probs is not None and len(best_pd_dd_labels) > 0:
                plot_roc_curves(
                    best_pd_dd_labels, best_pd_dd_preds, best_pd_dd_probs,
                    f"{plot_dir}/roc_pd_vs_dd.png"
                )
            
            if fold_features is not None:
                plot_tsne(fold_features, fold_hc_pd_labels, fold_pd_dd_labels, output_dir=plot_dir)
    
    # Print summary
    print("\n" + "="*60)
    print("ABLATION STUDY COMPLETE")
    print("="*60)
    print(f"Ablation Type: {ablation_type}")
    print(f"Folds trained: {len(all_fold_results)}")
    
    accuracies = [r['best_val_accuracy'] for r in all_fold_results]
    print(f"Best Val Accuracy per fold: {[f'{a:.4f}' for a in accuracies]}")
    print(f"Mean Accuracy: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
    
    return all_fold_results


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main function for TimesFM ablation studies with checkpoint resumption."""
    
    # Base configuration
    base_config = {
        'data_root': "/kaggle/input/datasets/meherujannat/parkinsons/pads-parkinsons-disease-smartwatch-dataset-1.0.0",
        'apply_downsampling': True,
        'apply_bandpass_filter': True,
        
        # Model settings
        'model_dim': 1280,  # TimesFM hidden size
        'hidden_dim': 512,
        'dropout': 0.3,
        'label_smoothing': 0.1,
        
        # LoRA settings (for lora ablation)
        'lora_r': 8,
        'lora_alpha': 16,
        'lora_dropout': 0.15,
        
        # Training settings
        'num_folds': 5,
        'max_folds_to_train': 1,
        'batch_size': 32,
        'learning_rate': 1e-4,
        'weight_decay': 1e-2,
        'num_epochs': 40,
        'early_stop_patience': 10,
        'num_workers': 0,
        
        # Output settings
        'save_metrics': True,
        'create_plots': True,
    }
    
    # Choose ablation type: 'gradual_unfreeze' or 'lora'
    ablation_type = 'lora'  
    
    # Per-ablation-type hyperparameter overrides
    if ablation_type == 'gradual_unfreeze':
        base_config['backbone_lr'] = 2e-5      
        base_config['head_lr'] = 1e-3           
        base_config['learning_rate'] = 2e-5     
        base_config['num_epochs'] = 50          
        base_config['warmup_epochs'] = 5        
        base_config['unfreeze_after_epoch'] = 10  
        base_config['unfreeze_n_layers'] = 2   
    elif ablation_type == 'lora':
        base_config['learning_rate'] = 2e-4   
        base_config['num_epochs'] = 40
        base_config['warmup_epochs'] = 3
    
    config = {**base_config}
    config['ablation_type'] = ablation_type
    config['output_dir'] = f'results/timesfm_{ablation_type}'
    
    resume_from = "/kaggle/input/models/meherujannat/loratimesfm/tensorflow2/default/1/best_model_fold_1.pth"
    
    print("\n" + "="*70)
    print("TimesFM ABLATION STUDY FOR PARKINSON'S DISEASE DETECTION")
    print("="*70)
    print(f"\nAblation Type: {ablation_type}")
    print(f"Epochs: {config['num_epochs']}")
    print(f"Folds: {config['max_folds_to_train']}/{config['num_folds']}")
    if 'backbone_lr' in config:
        print(f"Learning Rate: backbone={config['backbone_lr']}, heads={config['head_lr']}")
    else:
        print(f"Learning Rate: {config['learning_rate']}")
    print(f"Output Directory: {config['output_dir']}")
    
    if resume_from:
        print(f"\n*** RESUMING FROM CHECKPOINT ***")
        print(f"Checkpoint: {resume_from}")
    
    results = train_timesfm_model(config, resume_from_checkpoint=resume_from)
    
    return results


if __name__ == "__main__":
    results = main()