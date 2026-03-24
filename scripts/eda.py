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
import math  # ADD THIS
import csv   # ADD THIS

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
    

# ─────────────────────────────────────────────────────────────────────────────
# EDA Visualization Functions
# ─────────────────────────────────────────────────────────────────────────────

CHANNEL_NAMES = ['Acc X', 'Acc Y', 'Acc Z', 'Gyro X', 'Gyro Y', 'Gyro Z']

# Per-class colour scheme (used in all three plots)
CLASS_INFO = {
    'HC':  {'overlap': 0.70, 'color': '#27AE60', 'label': 'Healthy Control (HC)'},
    'PD':  {'overlap': 0.00, 'color': '#2980B9', 'label': "Parkinson's Disease (PD)"},
    'DD':  {'overlap': 0.65, 'color': '#C0392B', 'label': 'Dyskinetic Disorder (DD)'},
}

# Colours for individual IMU channels (used in patient-signals plot)
CHAN_COLORS = ['#E74C3C', '#3498DB', '#2ECC71', '#9B59B6', '#F39C12', '#1ABC9C']


def _style_white_ax(ax):
    """Apply clean white-background publication style to an axis."""
    ax.set_facecolor('white')
    for spine in ax.spines.values():
        spine.set_color('#AAAAAA')
        spine.set_linewidth(0.8)
    ax.tick_params(colors='#333333', labelsize=9)
    ax.xaxis.label.set_color('#333333')
    ax.yaxis.label.set_color('#333333')
    
def _load_raw_signal(data_root: str, patient_id: int, task: str = 'Relaxed'):
    """Load raw LeftWrist signal (no downsampling, no filter) for a given patient & task."""
    tmpl = pathlib.Path(data_root) / 'movement' / 'timeseries' / '{N:03d}_{X}_{Y}.txt'
    path = pathlib.Path(str(tmpl).format(N=patient_id, X=task, Y='LeftWrist'))
    if not path.exists():
        return None
    data = np.loadtxt(path, delimiter=',')
    # File layout: col 0 = timestamp, cols 1-6 = Acc XYZ + Gyro XYZ
    # Always strip the timestamp; fall back to first 6 cols if only 6 present
    if data.shape[1] >= 7:
        data = data[:, 1:7]   # skip timestamp, take the 6 sensor channels
    elif data.shape[1] > 6:
        data = data[:, :6]
    if data.shape[0] > 50:
        data = data[50:, :]
    return data


def _find_representative_patients(data_root: str, task: str = 'Relaxed') -> dict:
    """
    Scan patient files to find the first valid HC, PD, and DD patient that
    has data for the given task. Returns {class_key: patient_id}.
    """
    patients_tmpl   = pathlib.Path(data_root) / 'patients' / 'patient_{p:03d}.json'
    timeseries_tmpl = pathlib.Path(data_root) / 'movement' / 'timeseries' / '{N:03d}_{X}_{Y}.txt'

    found = {}
    for pid in range(1, 470):
        if len(found) == 3:
            break
        ppath = pathlib.Path(str(patients_tmpl).format(p=pid))
        if not ppath.exists():
            continue
        try:
            with open(ppath, 'r') as f:
                condition = json.load(f).get('condition', '')
        except Exception:
            continue

        if condition == 'Healthy':
            cls = 'HC'
        elif 'Parkinson' in condition:
            cls = 'PD'
        else:
            cls = 'DD'

        if cls in found:
            continue

        sig_path = pathlib.Path(str(timeseries_tmpl).format(N=pid, X=task, Y='LeftWrist'))
        if sig_path.exists():
            found[cls] = pid

    return found


# ── Plot 1: Patient Signals ───────────────────────────────────────────────────

def plot_patient_signals(data_root: str,
                         task: str = 'Relaxed',
                         save_path: str = 'eda_patient_signals.png'):
    """
    3×6 grid: rows = HC / PD / DD, columns = 6 IMU channels.
    White background, each channel in its own colour.
    """
    print("Finding representative patients ...")
    patient_map = _find_representative_patients(data_root, task)
    if len(patient_map) < 3:
        print(f"[WARNING] Only found {len(patient_map)} classes: {list(patient_map.keys())}")

    fig, axes = plt.subplots(3, 6, figsize=(22, 9), facecolor='white')
    fig.suptitle(
        f'Wrist IMU Signals per Class  —  Task: {task}',
        fontsize=15, color='#1A1A2E', fontweight='bold', y=1.01
    )

    for row_idx, cls in enumerate(['HC', 'PD', 'DD']):
        pid = patient_map.get(cls)
        meta = CLASS_INFO[cls]
        cls_color = meta['color']
        row_label = meta['label']

        if pid is None:
            for col in range(6):
                ax = axes[row_idx, col]
                _style_white_ax(ax)
                ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                        color='gray', transform=ax.transAxes)
            axes[row_idx, 0].set_ylabel(row_label, color=cls_color,
                                        fontsize=10, fontweight='bold', labelpad=10)
            continue

        raw    = _load_raw_signal(data_root, pid, task)
        raw_ds = downsample(raw)
        t      = np.arange(len(raw_ds)) / 64.0

        for col_idx in range(6):
            ax = axes[row_idx, col_idx]
            _style_white_ax(ax)
            ax.set_facecolor(cls_color + '0D')    # 5% class-colour tint

            ax.plot(t, raw_ds[:, col_idx], color=CHAN_COLORS[col_idx],
                    linewidth=0.85, alpha=0.9)

            if row_idx == 0:
                ax.set_title(CHANNEL_NAMES[col_idx], color=CHAN_COLORS[col_idx],
                             fontsize=10, fontweight='bold', pad=5)
            if col_idx == 0:
                ax.set_ylabel(f'{row_label}\n(P{pid:03d})', color=cls_color,
                              fontsize=9, fontweight='bold', labelpad=8)
            if row_idx == 2:
                ax.set_xlabel('Time (s)', fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"[Saved] {save_path}")
    plt.show()
    return fig


# ── Plot 2: Overlapping Windows ───────────────────────────────────────────────

def plot_overlapping_windows(data_root: str,
                             task: str = 'Relaxed',
                             window_size: int = 256,
                             n_windows_shown: int = 5,
                             channel: int = 0,
                             save_path: str = 'eda_overlapping_windows.png'):
    """
    For each class show the signal with coloured rectangular window boxes on top,
    matching the reference image style:
      - signal in a neutral colour
      - each window = coloured rectangle outline (solid for odd, dashed for even)
      - right-pointing arrow inside each window shows slide direction
      - double-headed arrow + label marks the step size between consecutive windows
    """
    print("Finding representative patients for window visualisation ...")
    patient_map = _find_representative_patients(data_root, task)

    WIN_COLORS = ['#C0392B', '#F39C12', '#2980B9', '#8E44AD', '#27AE60',
                  '#E67E22', '#16A085', '#D35400']
    WIN_LS     = ['-',       '--',      '-',       '--',      '-',
                  '--',      '-',       '--']

    fig, axes = plt.subplots(3, 1, figsize=(16, 11), facecolor='white')
    fig.suptitle(
        f'Overlapping Sliding Windows  —  Channel: {CHANNEL_NAMES[channel]}  |  '
        f'Window size: {window_size} samples  ({window_size/64:.1f} s @ 64 Hz)',
        fontsize=13, color='#1A1A2E', fontweight='bold'
    )

    for row_idx, cls in enumerate(['HC', 'PD', 'DD']):
        ax   = axes[row_idx]
        meta = CLASS_INFO[cls]
        cls_color = meta['color']
        overlap   = meta['overlap']
        step = int(window_size * (1 - overlap))

        _style_white_ax(ax)

        pid = patient_map.get(cls)
        if pid is None:
            ax.text(0.5, 0.5, 'No patient data available', ha='center', va='center',
                    color='gray', transform=ax.transAxes, fontsize=12)
            ax.set_ylabel(meta['label'], color=cls_color, fontsize=10, fontweight='bold')
            continue

        raw    = _load_raw_signal(data_root, pid, task)
        raw_ds = downsample(raw)

        starts   = list(range(0, len(raw_ds) - window_size + 1, step))[:n_windows_shown]
        view_end = min(starts[-1] + window_size + step // 2, len(raw_ds))

        t_all    = np.arange(view_end) / 64.0
        sig_view = raw_ds[:view_end, channel]

        ymin, ymax = float(sig_view.min()), float(sig_view.max())
        sig_range  = ymax - ymin if (ymax - ymin) > 0 else 1.0
        pad  = sig_range * 0.18
        ymin -= pad
        ymax += pad * 1.6     # extra headroom for labels above boxes

        # ── Background signal (thin, dark grey) ─────────────────────
        ax.plot(t_all, sig_view, color='#2C3E50', linewidth=1.1,
                alpha=0.45, zorder=1)

        # ── Window rectangles ────────────────────────────────────────
        box_bottom = ymin + pad * 0.55    # leave arrow space at the very bottom
        box_top    = ymax - pad * 0.45    # leave label space at the top

        for w_idx, start in enumerate(starts):
            end    = start + window_size
            wcolor = WIN_COLORS[w_idx % len(WIN_COLORS)]
            wls    = WIN_LS[w_idx % len(WIN_LS)]
            t0     = start / 64.0
            t1     = end   / 64.0
            t_mid  = (t0 + t1) / 2.0

            # Signal inside this window — highlighted in window colour
            ax.plot(t_all[start:end], sig_view[start:end],
                    color=wcolor, linewidth=1.6, alpha=0.92, zorder=3)

            # Rectangle outline (no fill — pure border like the reference image)
            rect = plt.matplotlib.patches.Rectangle(
                (t0, box_bottom),
                t1 - t0,
                box_top - box_bottom,
                linewidth=2.4,
                edgecolor=wcolor,
                facecolor='none',
                linestyle=wls,
                zorder=4,
                clip_on=False
            )
            ax.add_patch(rect)

            # Window label above box
            ax.text(t_mid, box_top + pad * 0.08,
                    f'W{w_idx + 1}', ha='center', va='bottom',
                    color=wcolor, fontsize=9, fontweight='bold', zorder=5)

            # Right-pointing arrow at the bottom inside the box
            arrow_y = box_bottom + pad * 0.12
            ax.annotate(
                '', xy=(t1 - 0.02, arrow_y), xytext=(t0 + 0.02, arrow_y),
                arrowprops=dict(arrowstyle='->', color=wcolor,
                                lw=1.8, mutation_scale=12),
                zorder=5
            )

        # ── Step-size brace (between W1 and W2 starts) ──────────────
        if len(starts) >= 2:
            s0_t = starts[0] / 64.0
            s1_t = starts[1] / 64.0
            brace_y = ymin + pad * 0.20
            ax.annotate(
                '', xy=(s1_t, brace_y), xytext=(s0_t, brace_y),
                arrowprops=dict(arrowstyle='<->', color='#444444', lw=1.3),
                zorder=6
            )
            ax.text((s0_t + s1_t) / 2, brace_y - pad * 0.22,
                    f'step = {step} samples  ({step/64:.2f} s)',
                    ha='center', va='top', fontsize=8,
                    color='#444444', fontstyle='italic')

        # ── Axis labels / title ──────────────────────────────────────
        overlap_pct = int(overlap * 100)
        ax.set_title(
            f'{meta["label"]}  (Patient {pid:03d})   '
            f'overlap = {overlap_pct}%,  step = {step} samples',
            color=cls_color, fontsize=10, fontweight='bold', loc='left', pad=4
        )
        ax.set_ylabel(CHANNEL_NAMES[channel], fontsize=9)
        ax.set_ylim(ymin, ymax)
        ax.set_xlim(t_all[0] - 0.05, t_all[-1] + 0.05)
        if row_idx == 2:
            ax.set_xlabel('Time (s)', fontsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"[Saved] {save_path}")
    plt.show()
    return fig


# ── Plot 3: Bandpass Filter Comparison ───────────────────────────────────────

def plot_bandpass_comparison(data_root: str,
                             patient_id=None,
                             task: str = 'Relaxed',
                             channel: int = 0,
                             save_path: str = 'eda_bandpass_comparison.png'):
    """
    Plot a side-by-side comparison of:
      - Raw (downsampled, no filter) vs Bandpass filtered signal
    Also shows the frequency spectrum (FFT) for both versions.
    """
    # Find a PD patient if none specified (most interesting for tremor freq)
    if patient_id is None:
        patient_map = _find_representative_patients(data_root, task)
        patient_id = patient_map.get('PD') or patient_map.get('HC') or 1

    raw = _load_raw_signal(data_root, patient_id, task)
    if raw is None:
        print(f"[ERROR] No signal found for patient {patient_id}, task {task}")
        return

    raw_ds = downsample(raw)                       # downsampled only
    filtered = bandpass_filter(raw_ds)             # bandpass applied

    Fs = 64                                        # Hz after downsampling
    t = np.arange(len(raw_ds)) / Fs
    sig_raw = raw_ds[:, channel]
    sig_filt = filtered[:, channel]

    # FFT
    N = len(sig_raw)
    freqs = np.fft.rfftfreq(N, d=1 / Fs)
    fft_raw  = np.abs(np.fft.rfft(sig_raw))  / N
    fft_filt = np.abs(np.fft.rfft(sig_filt)) / N

    # ── Figure layout: 2 rows × 2 cols ──
    fig = plt.figure(figsize=(18, 9), facecolor='#0F1117')
    gs  = fig.add_gridspec(2, 2, hspace=0.40, wspace=0.32,
                           left=0.07, right=0.97, top=0.90, bottom=0.08)

    ax_raw_time  = fig.add_subplot(gs[0, 0])
    ax_filt_time = fig.add_subplot(gs[0, 1])
    ax_raw_freq  = fig.add_subplot(gs[1, 0])
    ax_filt_freq = fig.add_subplot(gs[1, 1])

    fig.suptitle(
        f'Bandpass Filter Effect  |  Patient {patient_id:03d}  |  Task: {task}  |  Channel: {CHANNEL_NAMES[channel]}',
        fontsize=15, color='white', fontweight='bold'
    )

    _style_ax = lambda ax: (
        ax.__setattr__('_facecolor', '#1A1D24') or
        ax.set_facecolor('#1A1D24') or
        ax.tick_params(colors='#8892A4', labelsize=9) or
        [s.set_color('#2C2F3A') for s in ax.spines.values()]
    )

    for ax in [ax_raw_time, ax_filt_time, ax_raw_freq, ax_filt_freq]:
        _style_ax(ax)

    # ── Time domain ──
    ax_raw_time.plot(t, sig_raw, color='#E74C3C', linewidth=0.9, alpha=0.9)
    ax_raw_time.set_title('Raw Signal (no filter)', color='#E74C3C',
                          fontsize=12, fontweight='bold')
    ax_raw_time.set_xlabel('Time (s)', color='#8892A4', fontsize=9)
    ax_raw_time.set_ylabel(CHANNEL_NAMES[channel], color='#8892A4', fontsize=9)

    ax_filt_time.plot(t, sig_filt, color='#2ECC71', linewidth=0.9, alpha=0.9)
    ax_filt_time.set_title('Bandpass Filtered  (0.1 – 20 Hz)', color='#2ECC71',
                           fontsize=12, fontweight='bold')
    ax_filt_time.set_xlabel('Time (s)', color='#8892A4', fontsize=9)
    ax_filt_time.set_ylabel(CHANNEL_NAMES[channel], color='#8892A4', fontsize=9)

    # ── Frequency domain ──
    freq_mask = freqs <= 30        # show up to 30 Hz
    ax_raw_freq.plot(freqs[freq_mask], fft_raw[freq_mask],
                     color='#E74C3C', linewidth=1.2, alpha=0.9)
    ax_raw_freq.axvspan(0,   0.1, alpha=0.15, color='#F39C12', label='Removed by filter')
    ax_raw_freq.axvspan(20,  30,  alpha=0.15, color='#F39C12')
    ax_raw_freq.set_title('Frequency Spectrum — Raw',
                          color='#E74C3C', fontsize=12, fontweight='bold')
    ax_raw_freq.set_xlabel('Frequency (Hz)', color='#8892A4', fontsize=9)
    ax_raw_freq.set_ylabel('Amplitude', color='#8892A4', fontsize=9)
    ax_raw_freq.axvline(0.1, color='#F39C12', lw=1.4, linestyle='--', label='Filter cutoffs (0.1 / 20 Hz)')
    ax_raw_freq.axvline(20,  color='#F39C12', lw=1.4, linestyle='--')
    ax_raw_freq.legend(fontsize=8, facecolor='#262B36', labelcolor='white', framealpha=0.6)

    ax_filt_freq.plot(freqs[freq_mask], fft_filt[freq_mask],
                      color='#2ECC71', linewidth=1.2, alpha=0.9)
    ax_filt_freq.axvspan(0,  0.1,  alpha=0.15, color='#F39C12')
    ax_filt_freq.axvspan(20, 30,   alpha=0.15, color='#F39C12')
    ax_filt_freq.set_title('Frequency Spectrum — Bandpass Filtered',
                           color='#2ECC71', fontsize=12, fontweight='bold')
    ax_filt_freq.set_xlabel('Frequency (Hz)', color='#8892A4', fontsize=9)
    ax_filt_freq.set_ylabel('Amplitude', color='#8892A4', fontsize=9)
    ax_filt_freq.axvline(0.1, color='#F39C12', lw=1.4, linestyle='--')
    ax_filt_freq.axvline(20,  color='#F39C12', lw=1.4, linestyle='--')

    # Shade the bandpass passband on frequency plots
    for ax in [ax_raw_freq, ax_filt_freq]:
        ax.axvspan(0.1, 20, alpha=0.06, color='white', label='Passband')

    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#0F1117')
    print(f"[Saved] {save_path}")
    plt.show()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Entry point – run all EDA plots
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    DATA_ROOT = '/kaggle/input/parkinsons/pads-parkinsons-disease-smartwatch-dataset-1.0.0'
    TASK      = 'Relaxed'          # task to pull demo signals from
    OUT_DIR   = '.'                # where to save the plots

    import os
    os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 60)
    print(" EDA Plot 1 — Patient Signals (HC / PD / DD)")
    print("=" * 60)
    plot_patient_signals(
        data_root=DATA_ROOT,
        task=TASK,
        save_path=os.path.join(OUT_DIR, 'eda_patient_signals.png')
    )

    print("\n" + "=" * 60)
    print(" EDA Plot 2 — Overlapping Windows Visualisation")
    print("=" * 60)
    plot_overlapping_windows(
        data_root=DATA_ROOT,
        task=TASK,
        window_size=256,
        n_windows_shown=7,
        channel=0,                 # Acc X
        save_path=os.path.join(OUT_DIR, 'eda_overlapping_windows.png')
    )

    print("\n" + "=" * 60)
    print(" EDA Plot 3 — Bandpass Filter Comparison")
    print("=" * 60)
    plot_bandpass_comparison(
        data_root=DATA_ROOT,
        task=TASK,
        channel=0,                 # Acc X
        save_path=os.path.join(OUT_DIR, 'eda_bandpass_comparison.png')
    )

    print("\n✓ All EDA plots saved.")
