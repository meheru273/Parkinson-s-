import pathlib
import numpy as np
import json
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from typing import Dict, List, Tuple
import warnings
from scipy.signal import butter, filtfilt, resample_poly
from math import gcd
warnings.filterwarnings("ignore", category=UserWarning)

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
import os
import numpy as np
import json
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.manifold import TSNE
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
from collections import defaultdict
import os
import numpy as np
import warnings
import math
import csv

warnings.filterwarnings('ignore')
    
###############Helper functions##########
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


###############splitting methods################
def k_fold_split_method(data_root, full_dataset, k=5):
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
            labels=full_dataset.labels[train_mask],
            patient_ids=full_dataset.patient_ids[train_mask]
        )
        
        test_dataset = type(full_dataset)(
            data_root=None,
            left_samples=full_dataset.left_samples[test_mask],
            right_samples=full_dataset.right_samples[test_mask],
            labels=full_dataset.labels[test_mask],
            patient_ids=full_dataset.patient_ids[test_mask]
        )
        
        # Print fold info
        train_hc = np.sum(train_dataset.labels == 0)
        train_pd = np.sum(train_dataset.labels == 1)
        train_dd = np.sum(train_dataset.labels == 2)
        test_hc = np.sum(test_dataset.labels == 0)
        test_pd = np.sum(test_dataset.labels == 1)
        test_dd = np.sum(test_dataset.labels == 2)
        
        print(f"\nFold {fold_id+1}/{k}:")
        print(f"  Train: {len(train_dataset)} samples (HC={train_hc}, PD={train_pd}, DD={train_dd})")
        print(f"  Test:  {len(test_dataset)} samples (HC={test_hc}, PD={test_pd}, DD={test_dd})")
        
        fold_datasets.append((train_dataset, test_dataset))
    
    return fold_datasets

####################dataloader###################
class ThreeClassDataLoader(Dataset):
    """Dataset for three-class classification: HC (0), PD (1), DD (2)."""
    
    def __init__(self, data_root: str = None, window_size: int = 256, 
                 left_samples=None, right_samples=None, 
                 labels=None,
                 apply_dowsampling=True,
                 apply_bandpass_filter=True, **kwargs):
        
        self.left_samples = []
        self.right_samples = []
        self.labels = []        # Single label: 0=HC, 1=PD, 2=DD
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
            if labels is not None:
                self.labels = np.array(labels) if not isinstance(labels, np.ndarray) else labels
            
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
            
                # Three-class labels: HC=0, PD=1, DD=2
                if condition == 'Healthy':
                    label = 0   # Healthy Control
                    overlap = 0.72      # overlap for differential sampling 
                elif 'Parkinson' in condition:
                    label = 1   # Parkinson's Disease
                    overlap = 0         # overlap for differential sampling
                else:  
                    label = 2   # Differential Diagnosis (other disorders)
                    overlap = 0.65      # overlap for differential sampling

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
                        self.labels.append(label)
                        self.patient_ids.append(patient_id)
                        self.task_names.append(patient_task_names[i])
                
            except Exception as e:
                print(f"Error loading patient {patient_id}: {e}")
                continue
        
        self.left_samples = np.array(self.left_samples)
        self.right_samples = np.array(self.right_samples)
        self.labels = np.array(self.labels)
        self.patient_ids = np.array(self.patient_ids)
        self.task_names = np.array(self.task_names)

        print(f"\nLoaded {len(self.labels)} total samples: "
              f"HC={np.sum(self.labels == 0)}, "
              f"PD={np.sum(self.labels == 1)}, "
              f"DD={np.sum(self.labels == 2)}")


    def get_train_test_split(self, split_type=1, **kwargs):
            
        if split_type == 3:
            # K-fold split (patient-level)
            k = kwargs.get('k', 10)
            
            if self.data_root is None:
                raise ValueError("data_root is required for K-fold split")
            
            fold_datasets = k_fold_split_method(self.data_root, self, k)
            
            return fold_datasets
        
        else:
            raise ValueError(f"Invalid split_type: {split_type}. Use 3 (k-fold)")


    def __len__(self):
        return len(self.left_samples) if hasattr(self, 'left_samples') and isinstance(self.left_samples, (list, np.ndarray)) else 0
    
    def __getitem__(self, idx):
        left_sample = torch.FloatTensor(self.left_samples[idx])
        right_sample = torch.FloatTensor(self.right_samples[idx])
        label = torch.LongTensor([self.labels[idx]]).squeeze()
        
        return left_sample, right_sample, label
    
    
###################model##################
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


class ThreeClassModel(nn.Module):
    """Same backbone as MainModel but with a single 3-class classifier head (HC/PD/DD)."""
    
    def __init__(
        self,
        input_dim: int = 6,  
        model_dim: int = 128,
        num_heads: int = 8,
        num_layers: int = 4,
        d_ff: int = 512,
        dropout: float = 0.1,
        timestep: int = 256,
        num_classes: int = 3,
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

        # Single three-class classification head: HC(0) vs PD(1) vs DD(2)
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, model_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim, num_classes)
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
        
        fused_features = self.get_features(left_wrist, right_wrist, device)
        logits = self.classifier(fused_features)

        return logits
    

################EvaluationFunctions###########
CLASS_NAMES = ["HC", "PD", "DD"]

def calculate_metrics(y_true, y_pred, task_name="", verbose=True):
    if len(y_true) == 0:
        return {}
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
    precision_avg, recall_avg, f1_avg, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
    
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    
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
                label_name = CLASS_NAMES[label] if label < len(CLASS_NAMES) else f"Class_{label}"
                print(f"{label_name}: Precision={precision[i]:.4f}, Recall={recall[i]:.4f}, F1={f1[i]:.4f}, Support={support[i]}")
        
        print("Confusion Matrix (rows=true, cols=pred):")
        print(f"       {'  '.join(CLASS_NAMES)}")
        for i, row in enumerate(cm):
            print(f"  {CLASS_NAMES[i]}  {row}")
    
    return metrics

def save_fold_metric(fold_idx, fold_suffix, best_epoch, best_val_acc,
                     fold_metrics):

    os.makedirs("metrics_3class", exist_ok=True)

    # helper writer
    def write_csv(filename, metrics_list):
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                "epoch",
                "accuracy",
                # weighted averages
                "precision_avg", "recall_avg", "f1_avg",
                # per-class: HC (0), PD (1), DD (2)
                "precision_HC", "recall_HC", "f1_HC",
                "precision_PD", "recall_PD", "f1_PD",
                "precision_DD", "recall_DD", "f1_DD",
            ])
            for epoch_data in metrics_list:
                m = epoch_data['metrics']
                prec_cls  = m.get('precision_per_class', [0, 0, 0])
                rec_cls   = m.get('recall_per_class',    [0, 0, 0])
                f1_cls    = m.get('f1_per_class',        [0, 0, 0])
                # safely index – guard against fewer classes in a fold
                def _get(arr, i): return float(arr[i]) if i < len(arr) else 0.0
                writer.writerow([
                    epoch_data['epoch'],
                    m.get('accuracy', 0),
                    # weighted averages
                    m.get('precision_avg', 0),
                    m.get('recall_avg', 0),
                    m.get('f1_avg', 0),
                    # HC
                    _get(prec_cls, 0), _get(rec_cls, 0), _get(f1_cls, 0),
                    # PD
                    _get(prec_cls, 1), _get(rec_cls, 1), _get(f1_cls, 1),
                    # DD
                    _get(prec_cls, 2), _get(rec_cls, 2), _get(f1_cls, 2),
                ])

    if fold_metrics:
        filename = f"metrics_3class/three_class_metrics{fold_suffix}.csv"
        write_csv(filename, fold_metrics)
        print(f"✓ Three-class metrics saved: {filename}")


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
    

def plot_roc_curves(labels, probabilities, output_path):
    """Plot one-vs-rest ROC curves for 3-class classification."""
    labels_np = np.array(labels)
    probs_np = np.array(probabilities)
    
    # Binarize labels for one-vs-rest
    labels_bin = label_binarize(labels_np, classes=[0, 1, 2])
    
    plt.figure(figsize=(10, 8))
    colors = ['blue', 'red', 'green']
    
    for i, (cls_name, color) in enumerate(zip(CLASS_NAMES, colors)):
        if i < probs_np.shape[1]:
            fpr, tpr, _ = roc_curve(labels_bin[:, i], probs_np[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, color=color, lw=2, 
                     label=f'{cls_name} (AUC = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves (One-vs-Rest): HC / PD / DD', fontsize=14)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_tsne(features, labels, output_dir="plots"):
    
    if features is None or len(features) == 0:
        print("No features available for t-SNE visualization")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("Performing t-SNE dimensionality reduction...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    features_2d = tsne.fit_transform(features)

    plt.figure(figsize=(10, 8))
    
    colors = ['blue', 'red', 'green']
    for cls_idx, (cls_name, color) in enumerate(zip(CLASS_NAMES, colors)):
        mask = labels == cls_idx
        if np.any(mask):
            plt.scatter(features_2d[mask, 0], features_2d[mask, 1], 
                        c=color, label=f'{cls_name} (n={np.sum(mask)})', 
                        alpha=0.6, s=50, edgecolors='black', linewidth=0.5)

    plt.title("t-SNE: HC vs PD vs DD")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, "tsne_three_class.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("[saved] tsne_three_class.png")

    return features_2d


###################trainer################ 

def train_single_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    train_loss = 0.0
    all_preds, all_labels = [], []
    
    for batch in tqdm(dataloader, desc="Training"):
        left_sample, right_sample, labels = batch
        
        left_sample = left_sample.to(device)
        right_sample = right_sample.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        logits = model(left_sample, right_sample, device) 
        
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        train_loss += loss.item()
        
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    train_loss /= len(dataloader)
    
    # Calculate training metrics
    train_metrics = calculate_metrics(all_labels, all_preds, 
                                      "Training HC vs PD vs DD", verbose=False)
    
    return train_loss, train_metrics


def validate_single_epoch(model, dataloader, criterion, device):
    """Validate for one epoch"""
    model.eval()
    val_loss = 0.0
    all_preds, all_labels, all_probs = [], [], []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            left_sample, right_sample, labels = batch
            
            left_sample = left_sample.to(device)
            right_sample = right_sample.to(device)
            labels = labels.to(device)
            
            logits = model(left_sample, right_sample, device)  
            
            loss = criterion(logits, labels)
            val_loss += loss.item()
            
            preds = torch.argmax(logits, dim=1)
            probs = F.softmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    val_loss /= len(dataloader)
    
    return val_loss, all_preds, all_labels, all_probs

def extract_features(model, dataloader, device):
    model.eval()
    all_features = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting features"):
            left_sample, right_sample, labels = batch
            
            left_sample = left_sample.to(device)
            right_sample = right_sample.to(device)
            
            features = model.get_features(left_sample, right_sample, device) 
            
            all_features.append(features.cpu().numpy())
            all_labels.append(labels.numpy())
    
    all_features = np.vstack(all_features)
    all_labels = np.concatenate(all_labels)
    
    return all_features, all_labels

def train_model(config):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    output_dir = config.get('output_dir', 'output_3class')
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "metrics"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)
    
    # Save config.json once before training
    config_serializable = {k: v for k, v in config.items()
                           if isinstance(v, (str, int, float, bool, list, dict, type(None)))}
    with open(os.path.join(output_dir, 'checkpoints', 'config.json'), 'w') as f:
        json.dump(config_serializable, f, indent=2)
    print(f"✓ Config saved: {output_dir}/checkpoints/config.json")
    
    # Load dataset
    full_dataset = ThreeClassDataLoader(
        config['data_root'],
        apply_dowsampling=config['apply_downsampling'],
        apply_bandpass_filter=config['apply_bandpass_filter']
    )
    
    split_type = config.get('split_type', 3)
    
    if split_type == 3:
        fold_datasets = full_dataset.get_train_test_split(split_type=3, k=config['num_folds'])
        num_folds = len(fold_datasets)
    else:
        raise ValueError(f"Invalid split_type: {split_type}. Use 3 (k-fold)")
    
    all_fold_results = []
    
    max_folds = config.get('max_folds_to_train', num_folds)
    folds_to_train = min(num_folds, max_folds)
    print(f"Training {folds_to_train} out of {num_folds} folds")
    
    for fold_idx in range(folds_to_train):
        
        if num_folds > 1:
            print(f"\n{'='*60}")
            print(f"Starting Fold {fold_idx+1}/{num_folds}")
            print(f"{'='*60}")
        
        train_dataset, val_dataset = fold_datasets[fold_idx]
        
        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=config['num_workers'])
        
        val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers'])
        
        # Model
        model = ThreeClassModel(
            input_dim=config['input_dim'],
            model_dim=config['model_dim'],
            num_heads=config['num_heads'],
            num_layers=config['num_layers'],
            d_ff=config['d_ff'],
            dropout=config['dropout'],
            timestep=config['timestep'],
            num_classes=config['num_classes']
        ).to(device)
        
        optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        
        criterion = nn.CrossEntropyLoss()
        
        history = defaultdict(list)
        best_val_acc = 0.0
        best_epoch = 0
        fold_features = None
        fold_labels = None
        
        fold_metrics_list = []
        
        best_probs = None
        best_preds = None
        best_labels = None
        
        for epoch in range(config['num_epochs']):
            
            print(f"\nEpoch {epoch + 1}/{config['num_epochs']}")
            
            #############Training phase###########
            train_loss, train_metrics = train_single_epoch(
                model, train_loader, criterion, optimizer, device
            )
            
            ###########Validation phase############
            val_loss, val_preds, val_labels, val_probs = validate_single_epoch(
                model, val_loader, criterion, device
            )
            
            print("\n" + "="*60)
            val_metrics = calculate_metrics(
                val_labels, val_preds,
                f"{'Fold ' + str(fold_idx+1) + ' ' if num_folds > 1 else ''}Validation HC vs PD vs DD",
                verbose=True
            )
            print("="*60)
            
            if val_labels:
                fold_metrics_list.append({
                    'epoch': epoch + 1,
                    'predictions': list(val_preds),
                    'labels': list(val_labels),
                    'metrics': val_metrics
                })
            
            val_acc = val_metrics.get('accuracy', 0)
            train_acc = train_metrics.get('accuracy', 0)
            
            scheduler.step(val_loss)
            
            # Save history
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            print(f"\n{'Fold ' + str(fold_idx+1) + ', ' if num_folds > 1 else ''}Epoch {epoch+1} Summary:")
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            print(f"Val Acc: {val_acc:.4f}")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch + 1
                
                # Store best predictions and probabilities for ROC curves
                if val_probs:
                    best_probs = np.array(val_probs)
                    best_preds = np.array(val_preds)
                    best_labels = np.array(val_labels)
                
                model_save_name = os.path.join(output_dir, 'checkpoints',
                    f'best_model{"_fold_" + str(fold_idx+1) if num_folds > 1 else ""}.pth')
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'fold': fold_idx if num_folds > 1 else None,
                    'epoch': epoch,
                    'val_acc': val_acc,
                    'config': config
                }, model_save_name)
                print(f"✓ New best model saved: {model_save_name}")
        
        if config.get('save_metrics', True):
            fold_suffix = f"_fold_{fold_idx+1}" if num_folds > 1 else ""
            
            if fold_metrics_list:
                save_fold_metric(fold_idx, fold_suffix, best_epoch, best_val_acc,
                               fold_metrics_list)
                
        fold_features, fold_labels = extract_features(
            model, val_loader, device
        )
        
        fold_result = {
            'best_val_accuracy': best_val_acc,
            'history': history,
            'features': fold_features,
            'labels': fold_labels
        }
        all_fold_results.append(fold_result)
        
        if config.get('create_plots', True):
            plot_dir = os.path.join(output_dir, 
                f"plots/{'fold_' + str(fold_idx+1) if num_folds > 1 else 'single_run'}")
            os.makedirs(plot_dir, exist_ok=True)
            
            plot_loss(history['train_loss'], history['val_loss'], f"{plot_dir}/loss.png")
            
            if best_probs is not None and len(best_labels) > 0:
                plot_roc_curves(best_labels, best_probs, f"{plot_dir}/roc_three_class.png")
            
            if fold_features is not None:
                plot_tsne(fold_features, fold_labels, output_dir=plot_dir)
    
    return all_fold_results


def main():
    """Main function with configurable parameters.
    Set USE_NESTED_CV = True  for nested cross-validation hyperparameters.
    Set USE_NESTED_CV = False for initial cross-validation hyperparameters.
    """
    
    USE_NESTED_CV = False   # <-- Toggle between initial CV and nested CV hyperparameters
    
    # ---------- shared settings ----------
    base_config = {
        'data_root': "/kaggle/input/parkinsons/pads-parkinsons-disease-smartwatch-dataset-1.0.0",
        'apply_downsampling': True,  # Set to False to skip downsampling (100 Hz → 64 Hz)
        'apply_bandpass_filter': True,
        'split_type': 3,
        'split_ratio': 0.85,
        'train_tasks': None,
        'num_folds': 5,
        'input_dim': 6,
        'timestep': 256,
        'num_classes': 3,       # Three classes: HC, PD, DD
        'num_epochs': 100,
        'num_workers': 0,
        'save_metrics': True,
        'create_plots': True,
        'max_folds_to_train': 1,
        'output_dir': 'output_3class',
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
    results = train_model(config)
    
    return results


if __name__ == "__main__":
    results = main()
