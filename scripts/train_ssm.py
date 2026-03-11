
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import defaultdict
import pathlib
import json
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from typing import Dict, List, Tuple
import warnings
from scipy.signal import butter, filtfilt
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

# =============================================================================
# Dataset
# =============================================================================
    
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
    step = int(original_freq // target_freq)  
    if step > 1:
        return data[::step, :]
    return data


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
####################dataloader###################
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
            # K-fold split (patient-level)
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
    


# =============================================================================
# Metrics
# =============================================================================

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

def save_fold_metric(fold_idx, fold_suffix, best_epoch, best_val_acc,
                     fold_metrics_hc, fold_metrics_pd, metrics_dir="metrics"):

    os.makedirs(metrics_dir, exist_ok=True)

    def write_csv(filename, metrics_list, class_names):
        """Write comprehensive per-epoch metrics CSV."""
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            # Build header with per-class columns
            header = ["epoch", "accuracy"]
            for cls in class_names:
                header += [f"precision_{cls}", f"recall_{cls}", f"f1_{cls}", f"support_{cls}"]
            header += ["precision_avg", "recall_avg", "f1_avg", "confusion_matrix"]
            writer.writerow(header)

            for epoch_data in metrics_list:
                m = epoch_data['metrics']
                row = [epoch_data['epoch'], m.get('accuracy', 0)]
                prec = m.get('precision_per_class', [])
                rec = m.get('recall_per_class', [])
                f1 = m.get('f1_per_class', [])
                sup = m.get('support_per_class', [])
                for i in range(len(class_names)):
                    row.append(prec[i] if i < len(prec) else 0)
                    row.append(rec[i] if i < len(rec) else 0)
                    row.append(f1[i] if i < len(f1) else 0)
                    row.append(sup[i] if i < len(sup) else 0)
                row.append(m.get('precision_avg', 0))
                row.append(m.get('recall_avg', 0))
                row.append(m.get('f1_avg', 0))
                cm = m.get('confusion_matrix', None)
                row.append(str(cm.tolist()) if cm is not None else "[]")
                writer.writerow(row)

    def write_summary(filename, best_epoch_data, class_names, best_epoch, best_val_acc):
        """Write a single-row summary CSV for the best epoch."""
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            header = ["best_epoch", "best_val_acc", "accuracy"]
            for cls in class_names:
                header += [f"precision_{cls}", f"recall_{cls}", f"f1_{cls}", f"support_{cls}"]
            header += ["precision_avg", "recall_avg", "f1_avg", "confusion_matrix"]
            writer.writerow(header)

            m = best_epoch_data['metrics']
            row = [best_epoch, best_val_acc, m.get('accuracy', 0)]
            prec = m.get('precision_per_class', [])
            rec = m.get('recall_per_class', [])
            f1 = m.get('f1_per_class', [])
            sup = m.get('support_per_class', [])
            for i in range(len(class_names)):
                row.append(prec[i] if i < len(prec) else 0)
                row.append(rec[i] if i < len(rec) else 0)
                row.append(f1[i] if i < len(f1) else 0)
                row.append(sup[i] if i < len(sup) else 0)
            row.append(m.get('precision_avg', 0))
            row.append(m.get('recall_avg', 0))
            row.append(m.get('f1_avg', 0))
            cm = m.get('confusion_matrix', None)
            row.append(str(cm.tolist()) if cm is not None else "[]")
            writer.writerow(row)

    # HC vs PD
    hc_classes = ["HC", "PD"]
    if fold_metrics_hc:
        hc_filename = os.path.join(metrics_dir, f"hc_vs_pd_metrics{fold_suffix}.csv")
        write_csv(hc_filename, fold_metrics_hc, hc_classes)
        print(f"✓ HC vs PD metrics saved: {hc_filename}")
        # Find best epoch entry and save summary
        best_hc = [e for e in fold_metrics_hc if e['epoch'] == best_epoch]
        if best_hc:
            summary_file = os.path.join(metrics_dir, f"hc_vs_pd_final_summary{fold_suffix}.csv")
            write_summary(summary_file, best_hc[0], hc_classes, best_epoch, best_val_acc)
            print(f"✓ HC vs PD summary saved: {summary_file}")

    # PD vs DD
    pd_classes = ["PD", "DD"]
    if fold_metrics_pd:
        pd_filename = os.path.join(metrics_dir, f"pd_vs_dd_metrics{fold_suffix}.csv")
        write_csv(pd_filename, fold_metrics_pd, pd_classes)
        print(f"✓ PD vs DD metrics saved: {pd_filename}")
        best_pd = [e for e in fold_metrics_pd if e['epoch'] == best_epoch]
        if best_pd:
            summary_file = os.path.join(metrics_dir, f"pd_vs_dd_final_summary{fold_suffix}.csv")
            write_summary(summary_file, best_pd[0], pd_classes, best_epoch, best_val_acc)
            print(f"✓ PD vs DD summary saved: {summary_file}")


def plot_loss(train_losses, val_losses, output_path):  # FIXED: Changed signature
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
    
    print("Performing t-SNE dimensionality reduction...")
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
        print("[saved] tsne_pd_vs_dd.png")

    return features_2d


###################trainer################ 

def train_single_epoch(model, dataloader, criterion_hc, criterion_pd, optimizer, device):
    """Train for one epoch"""
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
        
        # Backward pass
        if loss_count > 0:
            avg_loss = total_loss / loss_count
            avg_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.25)
            optimizer.step()
            train_loss += avg_loss.item()
    
    train_loss /= len(dataloader)
    
    # Calculate training metrics
    train_metrics_hc = calculate_metrics(hc_pd_train_labels, hc_pd_train_pred, 
                                        "Training HC vs PD", verbose=False)
    train_metrics_pd = calculate_metrics(pd_dd_train_labels, pd_dd_train_pred, 
                                        "Training PD vs DD", verbose=False)
    
    return train_loss, train_metrics_hc, train_metrics_pd


def validate_single_epoch(model, dataloader, criterion_hc, criterion_pd, device):
    """Validate for one epoch"""
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
            
            # HC vs PD loss
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
            
            # PD vs DD loss
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


# =============================================================================
# Shared Components (reused from base model)
# =============================================================================

class PositionalEncoding(nn.Module):
    def __init__(self, model_dim: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, model_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, model_dim, 2).float() * (-math.log(10000.0) / model_dim)
        )
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


class CrossAttention(nn.Module):
    """Bidirectional cross-attention between two streams (from base model)."""

    def __init__(self, model_dim: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.cross_attention_1to2 = nn.MultiheadAttention(
            embed_dim=model_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.cross_attention_2to1 = nn.MultiheadAttention(
            embed_dim=model_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.self_attention_1 = nn.MultiheadAttention(
            embed_dim=model_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.self_attention_2 = nn.MultiheadAttention(
            embed_dim=model_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.norm_cross_1 = nn.LayerNorm(model_dim)
        self.norm_cross_2 = nn.LayerNorm(model_dim)
        self.norm_self_1 = nn.LayerNorm(model_dim)
        self.norm_self_2 = nn.LayerNorm(model_dim)
        self.feed_forward_1 = FeedForward(model_dim, d_ff, dropout)
        self.feed_forward_2 = FeedForward(model_dim, d_ff, dropout)

    def forward(self, channel_1, channel_2):
        # Cross attention
        c1_cross, _ = self.cross_attention_1to2(query=channel_1, key=channel_2, value=channel_2)
        c1_cross = self.norm_cross_1(channel_1 + c1_cross)
        c2_cross, _ = self.cross_attention_2to1(query=channel_2, key=channel_1, value=channel_1)
        c2_cross = self.norm_cross_2(channel_2 + c2_cross)
        # Self attention
        c1_self, _ = self.self_attention_1(query=c1_cross, key=c1_cross, value=c1_cross)
        c1_self = self.norm_self_1(c1_cross + c1_self)
        c2_self, _ = self.self_attention_2(query=c2_cross, key=c2_cross, value=c2_cross)
        c2_self = self.norm_self_2(c2_cross + c2_self)
        # FFN
        return self.feed_forward_1(c1_self), self.feed_forward_2(c2_self)


# =============================================================================
# Mamba / SSM Building Blocks
# =============================================================================

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
        Sequential selective scan implementation.

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
        chunk_size = 32  # gradient paths limited to 32 steps instead of 256

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


# =============================================================================
# Gated Fusion Module
# =============================================================================

class GatedFusion(nn.Module):
    """
    Learned gated fusion between two embedding streams.
    g = σ(W·[x_a; x_b] + b)
    output = g * x_a + (1 - g) * x_b
    """

    def __init__(self, dim: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid(),
        )

    def forward(self, x_a, x_b):
        g = self.gate(torch.cat([x_a, x_b], dim=-1))
        return g * x_a + (1 - g) * x_b


# =============================================================================
# Classifier Head Builder
# =============================================================================

def _make_classifier_head(input_dim, hidden_dim, num_classes, dropout):
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, num_classes),
    )


# =============================================================================
# Model 1: Pure SSM (replaces CrossAttention entirely)
# =============================================================================

class PureSSMModel(nn.Module):
    """
    Pure state-space model replacing all cross-attention layers.

    Architecture:
        Projection → PositionalEncoding → SSMStack(left) + SSMStack(right)
        → Pool → Concat → ClassifierHeads
    """

    def __init__(
        self,
        input_dim: int = 6,
        model_dim: int = 128,
        num_heads: int = 8,       # unused, kept for interface compatibility
        num_layers: int = 4,
        d_ff: int = 512,
        dropout: float = 0.1,
        timestep: int = 256,
        num_classes: int = 2,
        fusion_method: str = "concat",
        # SSM-specific
        ssm_d_state: int = 16,
        ssm_expand: int = 2,
        ssm_d_conv: int = 4,
    ):
        super().__init__()
        self.model_dim = model_dim
        self.timestep = timestep

        # Projections
        self.left_projection = nn.Linear(input_dim, model_dim)
        self.right_projection = nn.Linear(input_dim, model_dim)
        self.positional_encoding = PositionalEncoding(model_dim, max_len=timestep)

        # SSM stacks (one per wrist)
        self.left_ssm = SSMStack(
            num_layers, model_dim, d_ff, ssm_d_state, ssm_expand, ssm_d_conv, dropout
        )
        self.right_ssm = SSMStack(
            num_layers, model_dim, d_ff, ssm_d_state, ssm_expand, ssm_d_conv, dropout
        )

        self.pool = AttentionPool(model_dim)
        self.dropout = nn.Dropout(dropout)

        fusion_dim = model_dim * 2
        self.head_hc_vs_pd = _make_classifier_head(fusion_dim, model_dim, 2, dropout)
        self.head_pd_vs_dd = _make_classifier_head(fusion_dim, model_dim, 2, dropout)

    def get_features(self, left_wrist, right_wrist, device=None):
        left = self.dropout(self.left_projection(left_wrist))
        right = self.dropout(self.right_projection(right_wrist))

        left = self.left_ssm(left)
        right = self.right_ssm(right)

        left_pool = self.pool(left)
        right_pool = self.pool(right)

        return torch.cat([left_pool, right_pool], dim=1)

    def forward(self, left_wrist, right_wrist, device=None):
        fused = self.get_features(left_wrist, right_wrist, device)
        return self.head_hc_vs_pd(fused), self.head_pd_vs_dd(fused)


# =============================================================================
# SSM-Augmented Encoding (Gated PE + SSM Context)
# =============================================================================

class SSMAugmentedEncoding(nn.Module):
    """
    Blends fixed sinusoidal PE with learned SSM temporal context into
    a single positional signal, gated onto the raw projection.

    output = x + sigmoid(gate) * blend_proj([PE, SSM(x)])

    The gate starts near zero so early training is dominated by the
    raw projection; as training progresses the model learns how much
    SSM context to inject.
    """

    def __init__(self, model_dim, max_len, ssm_stack):
        super().__init__()
        self.ssm = ssm_stack
        self.pe = PositionalEncoding(model_dim, max_len)
        self.gate = nn.Parameter(torch.tensor(0.1))
        self.blend_proj = nn.Linear(model_dim * 2, model_dim)

    def forward(self, x):
        # x: raw projection output (B, L, model_dim)
        pe_signal = self.pe(torch.zeros_like(x))       # (B, L, model_dim)
        ssm_context = self.ssm(x)                      # (B, L, model_dim)
        blended_pe = self.blend_proj(
            torch.cat([pe_signal, ssm_context], dim=-1)
        )                                              # (B, L, model_dim)
        return x + blended_pe * torch.sigmoid(self.gate)


# =============================================================================
# Model 2: SSMAugmentedEncoding → CrossAttention
# =============================================================================

class SSMEncoderModel(nn.Module):
    """
    Gated SSM-augmented encoding feeds into CrossAttention for
    inter-limb interaction.

    Architecture:
        Projection → SSMAugmentedEncoding(left/right) → CrossAttention
        → Pool → Concat → ClassifierHeads
    """

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
        fusion_method: str = "concat",
        # SSM-specific
        ssm_d_state: int = 16,
        ssm_expand: int = 2,
        ssm_d_conv: int = 4,
        ssm_num_layers: int = None,  # defaults to num_layers if None
        ca_num_layers: int = None,   # defaults to num_layers if None
    ):
        super().__init__()
        self.model_dim = model_dim
        self.timestep = timestep

        ssm_layers = ssm_num_layers if ssm_num_layers is not None else num_layers
        ca_layers = ca_num_layers if ca_num_layers is not None else num_layers

        # Projections
        self.left_projection = nn.Linear(input_dim, model_dim)
        self.right_projection = nn.Linear(input_dim, model_dim)

        # SSM-Augmented Encoding (one per wrist)
        self.left_ssm_aug = SSMAugmentedEncoding(
            model_dim, max_len=timestep,
            ssm_stack=SSMStack(ssm_layers, model_dim, d_ff, ssm_d_state, ssm_expand, ssm_d_conv, dropout),
        )
        self.right_ssm_aug = SSMAugmentedEncoding(
            model_dim, max_len=timestep,
            ssm_stack=SSMStack(ssm_layers, model_dim, d_ff, ssm_d_state, ssm_expand, ssm_d_conv, dropout),
        )

        # CrossAttention layers
        self.ca_layers = nn.ModuleList([
            CrossAttention(model_dim, num_heads, d_ff, dropout)
            for _ in range(ca_layers)
        ])

        self.pool = AttentionPool(model_dim)
        self.dropout = nn.Dropout(dropout)

        fusion_dim = model_dim * 2
        self.head_hc_vs_pd = _make_classifier_head(fusion_dim, model_dim, 2, dropout)
        self.head_pd_vs_dd = _make_classifier_head(fusion_dim, model_dim, 2, dropout)

    def get_features(self, left_wrist, right_wrist, device=None):
        left = self.dropout(self.left_projection(left_wrist))
        right = self.dropout(self.right_projection(right_wrist))

        # SSM-Augmented Encoding (blends PE + SSM context onto projection)
        left = self.left_ssm_aug(left)
        right = self.right_ssm_aug(right)

        # Cross-attention interaction
        for layer in self.ca_layers:
            left, right = layer(left, right)

        left_pool = self.pool(left)
        right_pool = self.pool(right)

        return torch.cat([left_pool, right_pool], dim=1)

    def forward(self, left_wrist, right_wrist, device=None):
        fused = self.get_features(left_wrist, right_wrist, device)
        return self.head_hc_vs_pd(fused), self.head_pd_vs_dd(fused)


# =============================================================================
# Model 3: Parallel CrossAttention + SSM with Gated Fusion
# =============================================================================

class GatedFusionModel(nn.Module):
    """
    Parallel dual-path model with learned gated fusion.

    Architecture:
        Projection
           ├── CrossAttention stack → Pool → left_ca, right_ca
           └── SSM stack            → Pool → left_ssm, right_ssm
                ↓
           GatedFusion(left_ca, left_ssm), GatedFusion(right_ca, right_ssm)
                ↓
           Concat(left_fused, right_fused)
                ↓
           ClassifierHeads
    """

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
        fusion_method: str = "concat",
        # SSM-specific
        ssm_d_state: int = 16,
        ssm_expand: int = 2,
        ssm_d_conv: int = 4,
    ):
        super().__init__()
        self.model_dim = model_dim
        self.timestep = timestep

        # Shared projections (input is shared, then forks into two paths)
        self.left_projection = nn.Linear(input_dim, model_dim)
        self.right_projection = nn.Linear(input_dim, model_dim)
        self.positional_encoding = PositionalEncoding(model_dim, max_len=timestep)

        # Path A: CrossAttention stack
        self.ca_layers = nn.ModuleList([
            CrossAttention(model_dim, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        # Path B: SSM stacks (one per wrist)
        self.left_ssm = SSMStack(
            num_layers, model_dim, d_ff, ssm_d_state, ssm_expand, ssm_d_conv, dropout
        )
        self.right_ssm = SSMStack(
            num_layers, model_dim, d_ff, ssm_d_state, ssm_expand, ssm_d_conv, dropout
        )

        self.pool = AttentionPool(model_dim)
        self.dropout = nn.Dropout(dropout)

        # Gated fusion (one per wrist)
        self.left_gate = GatedFusion(model_dim)
        self.right_gate = GatedFusion(model_dim)

        fusion_dim = model_dim * 2
        self.head_hc_vs_pd = _make_classifier_head(fusion_dim, model_dim, 2, dropout)
        self.head_pd_vs_dd = _make_classifier_head(fusion_dim, model_dim, 2, dropout)

    def get_features(self, left_wrist, right_wrist, device=None):
        # We need both Position Encoded paths (for CA) and non-PE paths (for SSM)
        left_proj = self.left_projection(left_wrist)
        right_proj = self.right_projection(right_wrist)

        left_pe = self.dropout(self.positional_encoding(left_proj))
        right_pe = self.dropout(self.positional_encoding(right_proj))
        
        left_no_pe = self.dropout(left_proj)
        right_no_pe = self.dropout(right_proj)

        # ---- Path A: CrossAttention ----
        left_ca, right_ca = left_pe, right_pe
        for layer in self.ca_layers:
            left_ca, right_ca = layer(left_ca, right_ca)
        left_ca_pool = self.pool(left_ca)
        right_ca_pool = self.pool(right_ca)

        # ---- Path B: SSM ----
        left_ssm = self.left_ssm(left_no_pe)
        right_ssm = self.right_ssm(right_no_pe)
        left_ssm_pool = self.pool(left_ssm)
        right_ssm_pool = self.pool(right_ssm)

        # ---- Gated fusion ----
        left_fused = self.left_gate(left_ca_pool, left_ssm_pool)
        right_fused = self.right_gate(right_ca_pool, right_ssm_pool)

        return torch.cat([left_fused, right_fused], dim=1)

    def forward(self, left_wrist, right_wrist, device=None):
        fused = self.get_features(left_wrist, right_wrist, device)
        return self.head_hc_vs_pd(fused), self.head_pd_vs_dd(fused)


# =============================================================================
# Model factory
# =============================================================================

MODEL_REGISTRY = {
    "pure_ssm": PureSSMModel,
    "ssm_encoder": SSMEncoderModel,
    "gated_fusion": GatedFusionModel,
}


def build_model(config):
    """Build the selected SSM model from config."""
    model_type = config["model_type"]
    ModelClass = MODEL_REGISTRY[model_type]

    # Common args
    kwargs = dict(
        input_dim=config["input_dim"],
        model_dim=config["model_dim"],
        num_heads=config["num_heads"],
        num_layers=config["num_layers"],
        d_ff=config["d_ff"],
        dropout=config["dropout"],
        timestep=config["timestep"],
        num_classes=config["num_classes"],
        # SSM-specific
        ssm_d_state=config.get("ssm_d_state", 16),
        ssm_expand=config.get("ssm_expand", 2),
        ssm_d_conv=config.get("ssm_d_conv", 4),
    )

    # SSMEncoderModel accepts extra layer-count overrides
    if model_type == "ssm_encoder":
        kwargs["ssm_num_layers"] = config.get("ssm_num_layers", None)
        kwargs["ca_num_layers"] = config.get("ca_num_layers", None)

    return ModelClass(**kwargs)


# =============================================================================
# Training loop (mirrors base_model_code.train_model but uses build_model)
# =============================================================================

def train_model(config):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Model type: {config['model_type']}")

    # Create structured output directory
    output_dir = os.path.join(config.get("output_dir", "results/ssm_results"), config["model_type"])
    metrics_dir = os.path.join(output_dir, "metrics")
    plots_base_dir = os.path.join(output_dir, "plots")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(plots_base_dir, exist_ok=True)

    # Save config.json
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2, default=str)
    print(f"✓ Config saved: {config_path}")

    # Load dataset
    full_dataset = ParkinsonsDataLoader(
        config["data_root"],
        apply_dowsampling=config["apply_downsampling"],
        apply_bandpass_filter=config["apply_bandpass_filter"],
    )

    split_type = config.get("split_type", 3)

    if split_type == 3:
        fold_datasets = full_dataset.get_train_test_split(
            split_type=3, k=config["num_folds"]
        )
        num_folds = len(fold_datasets)
    else:
        train_dataset, val_dataset = full_dataset.get_train_test_split(
            split_type=split_type,
            split_ratio=config.get("split_ratio", 0.85),
            train_tasks=config.get("train_tasks", None),
        )
        fold_datasets = [(train_dataset, val_dataset)]
        num_folds = 1

    all_fold_results = []

    for fold_idx in range(num_folds):

        if num_folds > 1:
            print(f"\n{'='*60}")
            print(f"Starting Fold {fold_idx+1}/{num_folds}")
            print(f"{'='*60}")

        from sklearn.model_selection import StratifiedKFold
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_curve, auc
        
        train_dataset, val_dataset = fold_datasets[fold_idx]

        train_loader = DataLoader(
            train_dataset,
            batch_size=config["batch_size"],
            shuffle=True,
            num_workers=config["num_workers"],
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config["batch_size"],
            shuffle=False,
            num_workers=config["num_workers"],
        )

        # Build SSM model
        model = build_model(config).to(device)

        # Print parameter count
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")

        optimizer = optim.AdamW(
            model.parameters(),
            lr=config["learning_rate"],
            weight_decay=config["weight_decay"],
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config["num_epochs"],
            eta_min=config["learning_rate"] * 0.01
        )

        hc_pd_loss = nn.CrossEntropyLoss()
        pd_dd_loss = nn.CrossEntropyLoss()

        history = defaultdict(list)
        best_val_acc_hc = 0.0
        best_val_acc_pd = 0.0
        best_epoch_hc = 0
        best_epoch_pd = 0
        
        early_stop_patience = 15
        patience_counter_hc = 0
        patience_counter_pd = 0

        fold_metrics_hc = []
        fold_metrics_pd = []

        best_hc_pd_probs = None
        best_hc_pd_preds = None
        best_hc_pd_labels = None
        best_pd_dd_probs = None
        best_pd_dd_preds = None
        best_pd_dd_labels = None

        for epoch in range(config["num_epochs"]):

            print(f"\nEpoch {epoch + 1}/{config['num_epochs']}")

            # Training
            train_loss, train_metrics_hc, train_metrics_pd = train_single_epoch(
                model, train_loader, hc_pd_loss, pd_dd_loss, optimizer, device
            )

            # Validation
            val_results = validate_single_epoch(
                model, val_loader, hc_pd_loss, pd_dd_loss, device
            )
            (
                val_loss,
                hc_pd_val_pred,
                hc_pd_val_labels,
                hc_pd_val_probs,
                pd_dd_val_pred,
                pd_dd_val_labels,
                pd_dd_val_probs,
            ) = val_results

            print("\n" + "=" * 60)
            fold_prefix = f"Fold {fold_idx+1} " if num_folds > 1 else ""
            val_metrics_hc = calculate_metrics(
                hc_pd_val_labels,
                hc_pd_val_pred,
                f"{fold_prefix}Validation HC vs PD",
                verbose=True,
            )
            val_metrics_pd = calculate_metrics(
                pd_dd_val_labels,
                pd_dd_val_pred,
                f"{fold_prefix}Validation PD vs DD",
                verbose=True,
            )
            print("=" * 60)

            if hc_pd_val_labels:
                fold_metrics_hc.append(
                    {
                        "epoch": epoch + 1,
                        "predictions": hc_pd_val_pred.copy(),
                        "labels": hc_pd_val_labels.copy(),
                        "metrics": val_metrics_hc,
                    }
                )

            if pd_dd_val_labels:
                fold_metrics_pd.append(
                    {
                        "epoch": epoch + 1,
                        "predictions": pd_dd_val_pred.copy(),
                        "labels": pd_dd_val_labels.copy(),
                        "metrics": val_metrics_pd,
                    }
                )

            val_acc_hc = val_metrics_hc.get("accuracy", 0)
            val_acc_pd = val_metrics_pd.get("accuracy", 0)
            val_acc_combined = (val_acc_hc + val_acc_pd) / 2

            train_acc_hc = train_metrics_hc.get("accuracy", 0)
            train_acc_pd = train_metrics_pd.get("accuracy", 0)

            scheduler.step()  # Unconditional step every epoch for CosineAnnealingLR

            # History
            history["train_loss"].append(train_loss)
            history["train_acc_hc"].append(train_acc_hc)
            history["train_acc_pd"].append(train_acc_pd)
            history["val_loss"].append(val_loss)
            history["val_acc_hc"].append(val_acc_hc)
            history["val_acc_pd"].append(val_acc_pd)
            history["val_acc_combined"].append(val_acc_combined)

            print(
                f"\n{fold_prefix}Epoch {epoch+1} Summary:\n"
                f"Train Loss: {train_loss:.4f}\n"
                f"Train Acc - HC vs PD: {train_acc_hc:.4f}, PD vs DD: {train_acc_pd:.4f}\n"
                f"Val Loss: {val_loss:.4f}\n"
                f"Val Acc - HC vs PD: {val_acc_hc:.4f}, PD vs DD: {val_acc_pd:.4f}, "
                f"Combined: {val_acc_combined:.4f}"
            )

            # Save best HC vs PD
            if val_acc_hc > best_val_acc_hc:
                best_val_acc_hc = val_acc_hc
                best_epoch_hc = epoch + 1
                patience_counter_hc = 0

                if hc_pd_val_probs:
                    best_hc_pd_probs = np.array(hc_pd_val_probs)
                    best_hc_pd_preds = np.array(hc_pd_val_pred)
                    best_hc_pd_labels = np.array(hc_pd_val_labels)

                model_save_name_hc = os.path.join(
                    output_dir,
                    f"best_{config['model_type']}_hc_fold{fold_idx+1}.pth"
                )
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'epoch': epoch,
                    'val_acc_hc': val_acc_hc,
                    'config': config,
                }, model_save_name_hc)
                print(f"✓ New best HC model saved (acc={val_acc_hc:.4f})")

            else:
                patience_counter_hc += 1

            # Save best PD vs DD
            if val_acc_pd > best_val_acc_pd:
                best_val_acc_pd = val_acc_pd
                best_epoch_pd = epoch + 1
                patience_counter_pd = 0

                if pd_dd_val_probs:
                    best_pd_dd_probs = np.array(pd_dd_val_probs)
                    best_pd_dd_preds = np.array(pd_dd_val_pred)
                    best_pd_dd_labels = np.array(pd_dd_val_labels)

                model_save_name_pd = os.path.join(
                    output_dir,
                    f"best_{config['model_type']}_pd_fold{fold_idx+1}.pth"
                )
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'epoch': epoch,
                    'val_acc_pd': val_acc_pd,
                    'config': config,
                }, model_save_name_pd)
                print(f"✓ New best PD model saved (acc={val_acc_pd:.4f})")
            else:
                patience_counter_pd += 1

            # Early Stopping Check
            if patience_counter_hc >= early_stop_patience and patience_counter_pd >= early_stop_patience:
                print(f"\nEarly stopping at epoch {epoch+1}: both tasks stagnant for {early_stop_patience} epochs.")
                break

            # Save metrics incrementally every epoch (overwrites previous)
            if config.get("save_metrics", True) and (fold_metrics_hc or fold_metrics_pd):
                try:
                    fold_suffix = f"_fold_{fold_idx+1}" if num_folds > 1 else ""
                    fold_suffix = f"_{config['model_type']}{fold_suffix}"
                    save_fold_metric(
                        fold_idx, fold_suffix, best_epoch_hc, best_val_acc_hc, # Pass HC bests, though it's a bit of a compromise for the shared file
                        fold_metrics_hc, fold_metrics_pd,
                        metrics_dir=metrics_dir,
                    )
                except Exception as e:
                    print(f"Warning: Could not save incremental metrics: {e}")

        # Final metrics save (ensures latest state is persisted)
        if config.get("save_metrics", True):
            try:
                fold_suffix = f"_fold_{fold_idx+1}" if num_folds > 1 else ""
                fold_suffix = f"_{config['model_type']}{fold_suffix}"
                if fold_metrics_hc or fold_metrics_pd:
                    save_fold_metric(
                        fold_idx, fold_suffix, best_epoch_hc, best_val_acc_hc, # using hc as proxy for best_epoch/acc signature
                        fold_metrics_hc, fold_metrics_pd,
                        metrics_dir=metrics_dir,
                    )
            except Exception as e:
                print(f"Warning: Could not save final metrics: {e}")

        try:
            model_save_name_hc = os.path.join(
                output_dir,
                f"best_{config['model_type']}_hc_fold{fold_idx+1}.pth"
            )
            if os.path.exists(model_save_name_hc):
                checkpoint = torch.load(model_save_name_hc, map_location=device, weights_only=False)
                model.load_state_dict(checkpoint['model_state_dict'])
        except Exception as e:
            print(f"Warning: Could not load best HC model for feature extraction: {e}")

        try:
            fold_features, fold_hc_pd_labels, fold_pd_dd_labels = extract_features(
                model, val_loader, device
            )
        except Exception as e:
            print(f"Warning: Could not extract features for t-SNE: {e}")
            fold_features, fold_hc_pd_labels, fold_pd_dd_labels = None, None, None

        fold_result = {
            "best_val_accuracy_hc": best_val_acc_hc,
            "best_val_accuracy_pd": best_val_acc_pd,
            "history": history,
            "features": fold_features,
            "hc_pd_labels": fold_hc_pd_labels,
            "pd_dd_labels": fold_pd_dd_labels,
        }
        all_fold_results.append(fold_result)

        # Plots
        if config.get("create_plots", True):
            plot_dir = os.path.join(
                plots_base_dir,
                f"{'fold_' + str(fold_idx+1) if num_folds > 1 else 'single_run'}"
            )
            os.makedirs(plot_dir, exist_ok=True)

            try:
                plot_loss(
                    history["train_loss"], history["val_loss"], f"{plot_dir}/loss.png"
                )
                print(f"✓ Loss plot saved: {plot_dir}/loss.png")
            except Exception as e:
                print(f"Warning: Could not save loss plot: {e}")

            try:
                if best_hc_pd_probs is not None and len(best_hc_pd_labels) > 0:
                    plot_roc_curves(
                        best_hc_pd_labels,
                        best_hc_pd_preds,
                        best_hc_pd_probs,
                        f"{plot_dir}/roc_hc_vs_pd.png",
                    )
                    print(f"✓ ROC HC vs PD saved: {plot_dir}/roc_hc_vs_pd.png")
            except Exception as e:
                print(f"Warning: Could not save HC vs PD ROC plot: {e}")

            try:
                if best_pd_dd_probs is not None and len(best_pd_dd_labels) > 0:
                    plot_roc_curves(
                        best_pd_dd_labels,
                        best_pd_dd_preds,
                        best_pd_dd_probs,
                        f"{plot_dir}/roc_pd_vs_dd.png",
                    )
                    print(f"✓ ROC PD vs DD saved: {plot_dir}/roc_pd_vs_dd.png")
            except Exception as e:
                print(f"Warning: Could not save PD vs DD ROC plot: {e}")

            try:
                if fold_features is not None:
                    plot_tsne(
                        fold_features,
                        fold_hc_pd_labels,
                        fold_pd_dd_labels,
                        output_dir=plot_dir,
                    )
            except Exception as e:
                print(f"Warning: Could not save t-SNE plot: {e}")

    # Print summary across folds
    if num_folds > 1:
        accs_hc = [r["best_val_accuracy_hc"] for r in all_fold_results]
        accs_pd = [r["best_val_accuracy_pd"] for r in all_fold_results]
        print(f"\n{'='*60}")
        print(f"Cross-validation summary ({config['model_type']}):")
        for i, (acc_hc, acc_pd) in enumerate(zip(accs_hc, accs_pd)):
            print(f"  Fold {i+1} - HC vs PD: {acc_hc:.4f}, PD vs DD: {acc_pd:.4f}")
        print(f"  Mean HC vs PD: {np.mean(accs_hc):.4f} ± {np.std(accs_hc):.4f}")
        print(f"  Mean PD vs DD: {np.mean(accs_pd):.4f} ± {np.std(accs_pd):.4f}")
        print(f"{'='*60}")

    return all_fold_results


# =============================================================================
# Entrypoint
# =============================================================================

def main():
    """Run training for the selected SSM model variant."""

    config = {
        # Data
        "data_root": "/kaggle/input/parkinsons/pads-parkinsons-disease-smartwatch-dataset-1.0.0",
        "apply_downsampling": True,
        "apply_bandpass_filter": True,
        "split_type": 3,
        "split_ratio": 0.85,
        "train_tasks": None,
        "num_folds": 5,
        
        # Model architecture
        "model_type": "ssm_encoder",  # <-- change to "ssm_encoder" or "gated_fusion"
        "input_dim": 6,
        "model_dim": 32,
        "num_heads": 8,
        "num_layers": 3,
        "d_ff": 256,
        "dropout": 0.12281570220908891,
        "timestep": 640,
        "num_classes": 2,
        
        # SSM-specific hyperparameters
        "ssm_d_state": 16,    
        "ssm_expand": 2,      
        "ssm_d_conv": 4,      

        "ssm_num_layers": 4,
        "ca_num_layers": None,
        
        # Training
        "batch_size": 32,
        "learning_rate": 0.0001,
        "weight_decay": 0.00016228510005606125,
        "num_epochs": 15,
        "num_workers": 0,
        
        # Output
        "output_dir": "results/ssm_results",
        "save_metrics": True,
        "create_plots": True,
    }

    results = train_model(config)
    return results


if __name__ == "__main__":
    results = main()
