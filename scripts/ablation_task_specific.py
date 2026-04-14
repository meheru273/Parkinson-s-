"""
Ablation Study: Task-Specific Models using 1D CNN and LSTM
For Parkinson's Disease Detection

This script trains separate models for each task using:
1. 1D CNN architecture
2. LSTM architecture
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
from scipy.signal import butter, filtfilt, resample_poly
from math import gcd
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


# ============== 1D CNN Model ==============
class CNN1DModel(nn.Module):
    """1D Convolutional Neural Network for time series classification"""
    
    def __init__(self, input_channels=12, seq_length=256, num_classes=2, dropout=0.3):
        super().__init__()
        
        self.conv_blocks = nn.Sequential(
            # Block 1
            nn.Conv1d(input_channels, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),
            
            # Block 2
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),
            
            # Block 3
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),
            
            # Block 4
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        self.feature_dim = 256
        
        # Classification heads
        self.head_hc_vs_pd = nn.Sequential(
            nn.Linear(self.feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 2)
        )
        
        self.head_pd_vs_dd = nn.Sequential(
            nn.Linear(self.feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 2)
        )
    
    def get_features(self, x):
        # x: [batch, seq_len, channels] -> [batch, channels, seq_len]
        x = x.transpose(1, 2)
        features = self.conv_blocks(x)
        features = features.squeeze(-1)
        return features
    
    def forward(self, x):
        features = self.get_features(x)
        logits_hc_vs_pd = self.head_hc_vs_pd(features)
        logits_pd_vs_dd = self.head_pd_vs_dd(features)
        return logits_hc_vs_pd, logits_pd_vs_dd


# ============== LSTM Model ==============
class LSTMModel(nn.Module):
    """LSTM model for time series classification"""
    
    def __init__(self, input_channels=12, hidden_size=128, num_layers=2, 
                 num_classes=2, dropout=0.3, bidirectional=True):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        # Input projection
        self.input_proj = nn.Linear(input_channels, hidden_size)
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        self.layer_norm = nn.LayerNorm(hidden_size * self.num_directions)
        self.dropout = nn.Dropout(dropout)
        
        self.feature_dim = hidden_size * self.num_directions
        
        # Classification heads
        self.head_hc_vs_pd = nn.Sequential(
            nn.Linear(self.feature_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 2)
        )
        
        self.head_pd_vs_dd = nn.Sequential(
            nn.Linear(self.feature_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 2)
        )
    
    def get_features(self, x):
        # x: [batch, seq_len, channels]
        x = self.input_proj(x)
        
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Use last hidden states from both directions
        if self.bidirectional:
            # Concatenate last hidden states from forward and backward
            h_forward = h_n[-2, :, :]  # Last layer, forward
            h_backward = h_n[-1, :, :]  # Last layer, backward
            features = torch.cat([h_forward, h_backward], dim=1)
        else:
            features = h_n[-1, :, :]
        
        features = self.layer_norm(features)
        features = self.dropout(features)
        
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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


def train_task_model(config, task_name, model_type='cnn'):
    """Train a model for a specific task"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f"Training {model_type.upper()} model for task: {task_name}")
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
        
        # Create model
        if model_type == 'cnn':
            model = CNN1DModel(
                input_channels=config['input_channels'],
                seq_length=config['window_size'],
                dropout=config['dropout']
            ).to(device)
        elif model_type == 'lstm':
            model = LSTMModel(
                input_channels=config['input_channels'],
                hidden_size=config['hidden_size'],
                num_layers=config['num_lstm_layers'],
                dropout=config['dropout'],
                bidirectional=config['bidirectional']
            ).to(device)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        
        criterion_hc = nn.CrossEntropyLoss()
        criterion_pd = nn.CrossEntropyLoss()
        
        best_val_acc = 0.0
        best_metrics = None
        history = defaultdict(list)
        
        for epoch in range(config['num_epochs']):
            # Training
            train_loss, train_hc, train_pd = train_epoch(
                model, train_loader, criterion_hc, criterion_pd, optimizer, device
            )
            
            # Validation
            val_loss, val_hc, val_pd = validate_epoch(
                model, val_loader, criterion_hc, criterion_pd, device
            )
            
            scheduler.step(val_loss)
            
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
                
                # Save best model
                os.makedirs(f"checkpoints/{model_type}", exist_ok=True)
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'task': task_name,
                    'fold': fold_idx,
                    'best_acc': best_val_acc,
                    'config': config
                }, f"checkpoints/{model_type}/{task_name}_fold{fold_idx+1}.pth")
        
        fold_result = {
            'fold': fold_idx + 1,
            'best_val_acc': best_val_acc,
            'best_metrics': best_metrics,
            'history': dict(history)
        }
        all_fold_results.append(fold_result)
        
        print(f"Fold {fold_idx+1} Best Combined Accuracy: {best_val_acc:.4f}")
    
    return all_fold_results


def run_ablation_study(config):
    """Run ablation study for all tasks and both model types"""
    
    tasks = config.get('tasks', [
        "CrossArms", "DrinkGlas", "Entrainment", "HoldWeight", "LiftHold",
        "PointFinger", "Relaxed", "StretchHold", "TouchIndex", "TouchNose"
    ])
    
    model_types = config.get('model_types', ['cnn', 'lstm'])
    
    all_results = {}
    
    # Save config.json once before running
    os.makedirs("checkpoints", exist_ok=True)
    config_serializable = {k: v for k, v in config.items()
                           if isinstance(v, (str, int, float, bool, list, dict, type(None)))}
    with open('checkpoints/config.json', 'w') as f:
        json.dump(config_serializable, f, indent=2)
    print("✓ Config saved: checkpoints/config.json")
    
    for model_type in model_types:
        all_results[model_type] = {}
        
        for task in tasks:
            print(f"\n{'#'*60}")
            print(f"ABLATION: {model_type.upper()} - {task}")
            print(f"{'#'*60}")
            
            results = train_task_model(config, task, model_type)
            
            if results is not None:
                all_results[model_type][task] = results

                # Calculate average metrics across folds
                avg_acc = np.mean([r['best_val_acc'] for r in results])
                std_acc = np.std([r['best_val_acc'] for r in results])
                print(f"\n{model_type.upper()} - {task}: Avg Acc = {avg_acc:.4f} ± {std_acc:.4f}")

                # Save incrementally after each task so results survive a timeout
                save_ablation_summary(all_results, config)

    # Final save
    save_ablation_summary(all_results, config)
    
    return all_results


def save_ablation_summary(results, config):
    """Save ablation study summary to CSV"""
    
    os.makedirs("ablation_results", exist_ok=True)
    
    summary_file = "ablation_results/task_specific_summary.csv"
    
    with open(summary_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Model', 'Task', 'Fold', 'HC_vs_PD_Acc', 'PD_vs_DD_Acc', 'Combined_Acc'])
        
        for model_type, task_results in results.items():
            for task, fold_results in task_results.items():
                for fold_data in fold_results:
                    hc_acc = fold_data['best_metrics']['hc_vs_pd'].get('accuracy', 0)
                    pd_acc = fold_data['best_metrics']['pd_vs_dd'].get('accuracy', 0)
                    combined = fold_data['best_val_acc']
                    writer.writerow([model_type, task, fold_data['fold'], 
                                   f"{hc_acc:.4f}", f"{pd_acc:.4f}", f"{combined:.4f}"])
    
    print(f"\nSummary saved to: {summary_file}")
    
    # Also save aggregated results
    agg_file = "ablation_results/task_specific_aggregated.csv"
    
    with open(agg_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Model', 'Task', 'Avg_Combined_Acc', 'Std_Combined_Acc', 
                        'Avg_HC_Acc', 'Avg_PD_Acc'])
        
        for model_type, task_results in results.items():
            for task, fold_results in task_results.items():
                combined_accs = [r['best_val_acc'] for r in fold_results]
                hc_accs = [r['best_metrics']['hc_vs_pd'].get('accuracy', 0) for r in fold_results]
                pd_accs = [r['best_metrics']['pd_vs_dd'].get('accuracy', 0) for r in fold_results]
                
                writer.writerow([
                    model_type, task,
                    f"{np.mean(combined_accs):.4f}", f"{np.std(combined_accs):.4f}",
                    f"{np.mean(hc_accs):.4f}", f"{np.mean(pd_accs):.4f}"
                ])
    
    print(f"Aggregated results saved to: {agg_file}")


def plot_ablation_results(results, output_dir="ablation_results"):
    """Create visualization of ablation study results"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare data for plotting
    tasks = list(list(results.values())[0].keys())
    model_types = list(results.keys())
    
    # Bar chart comparing models across tasks
    fig, ax = plt.subplots(figsize=(14, 6))
    
    x = np.arange(len(tasks))
    width = 0.35
    
    for i, model_type in enumerate(model_types):
        means = []
        stds = []
        for task in tasks:
            if task in results[model_type]:
                accs = [r['best_val_acc'] for r in results[model_type][task]]
                means.append(np.mean(accs))
                stds.append(np.std(accs))
            else:
                means.append(0)
                stds.append(0)
        
        offset = (i - len(model_types)/2 + 0.5) * width
        bars = ax.bar(x + offset, means, width, label=model_type.upper(), yerr=stds, capsize=3)
    
    ax.set_ylabel('Combined Accuracy')
    ax.set_xlabel('Task')
    ax.set_title('Task-Specific Model Performance: CNN vs LSTM')
    ax.set_xticks(x)
    ax.set_xticklabels(tasks, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/task_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Comparison plot saved to: {output_dir}/task_comparison.png")


def main():
    """Main function for ablation study"""
    
    config = {
        # Data settings
        'data_root': "/kaggle/input/parkinsons/pads-parkinsons-disease-smartwatch-dataset-1.0.0",
        'apply_downsampling': True,  # Set to False to skip downsampling (100 Hz → 64 Hz)
        'apply_bandpass_filter': True,
        'window_size': 256,
        'input_channels': 12,  # 6 channels × 2 wrists
        
        # Model settings
        'hidden_size': 128,
        'num_lstm_layers': 2,
        'bidirectional': True,
        'dropout': 0.3,
        
        # Training settings
        'num_folds': 5,
        'batch_size': 32,
        'learning_rate': 0.001,
        'weight_decay': 1e-4,
        'num_epochs': 50,
        'num_workers': 0,
        
        # Ablation settings
        'tasks': ["CrossArms", "DrinkGlas", "Entrainment", "HoldWeight", "LiftHold",
                  "PointFinger", "Relaxed", "StretchHold", "TouchIndex", "TouchNose"],
        'model_types': ['cnn', 'lstm']
    }
    
    print("=" * 60)
    print("ABLATION STUDY: Task-Specific Models (1D CNN & LSTM)")
    print("=" * 60)
    
    results = run_ablation_study(config)
    
    # Generate plots
    plot_ablation_results(results)
    
    print("\n" + "=" * 60)
    print("ABLATION STUDY COMPLETE")
    print("=" * 60)
    
    return results


if __name__ == "__main__":
    results = main()
