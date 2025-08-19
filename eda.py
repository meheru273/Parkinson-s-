import pathlib
import numpy as np
import json
from scipy.signal import resample

from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import defaultdict
import logging
from typing import Dict, List, Tuple
import warnings
import torch.nn.functional as F
import math



class ParkinsonsDatasetSplitter:
    def __init__(self, main_path: str, patient_path: str):
        self.main_path = main_path
        self.patient_path = patient_path
        self.patients_template = pathlib.Path(patient_path) / "patient_{p:03d}.json"
        self.timeseries_template = pathlib.Path(main_path) / "timeseries/{N:03d}_{X}_{Y}.txt"
        
        # Tasks from dataset structure
        self.tasks = ["CrossArms", "DrinkGlas", "Entrainment", "HoldWeight", "LiftHold", 
                     "PointFinger", "Relaxed", "StretchHold", "TouchIndex", "TouchNose"]
        self.wrists = ["LeftWrist", "RightWrist"]
        self.frequencies = [32, 16, 4]
        
        # Dataset has 469 participants (001-469)
        self.patient_ids = list(range(1, 470))
        print(f"Dataset: {len(self.patient_ids)} patients (001-469)")
        
    
    def load_data(self, patient_id: int, task: str, wrist: str):
        """Load 6-channel data: [Accel_X, Accel_Y, Accel_Z, Gyro_X, Gyro_Y, Gyro_Z]"""
        data_path = pathlib.Path(str(self.timeseries_template).format(N=patient_id, X=task, Y=wrist))
        patient_path = pathlib.Path(str(self.patients_template).format(p=patient_id))
        
        if not data_path.exists() or not patient_path.exists():
            return None, None
        
        # Load 6-channel data (100Hz sampling)
        data = np.loadtxt(data_path, delimiter=",")
        with open(patient_path, "r") as f:
            metadata = json.load(f)
        
        # Verify 6 channels
        if data.shape[1] != 6:
            print(f"Warning: Expected 6 channels, got {data.shape[1]} for {patient_id:03d}_{task}_{wrist}")
        
        return data, metadata
    
    def create_windows(self, data, window_size=32):
        """Create non-overlapping windows from 6-channel data"""
        if data is None:
            return None
        
        n_samples, n_channels = data.shape
        n_windows = n_samples // window_size
        
        if n_windows == 0:
            return None
        
        # Trim data to fit exact windows
        trimmed_data = data[:n_windows * window_size]
        # Shape: (n_windows, window_size, 6_channels)
        windows = trimmed_data.reshape(n_windows, window_size, n_channels)
        
        return windows
    
    def ten_fold_split(self):
        """Ten-fold cross-validation with proper fold rotation"""
        n_patients = len(self.patient_ids)
        fold_size = n_patients // 10  # ~46 patients per fold
        
        # Create 10 folds
        folds = []
        for i in range(10):
            start_idx = i * fold_size
            end_idx = start_idx + fold_size if i < 9 else n_patients
            folds.append(self.patient_ids[start_idx:end_idx])
        
        print("Starting 10-fold cross-validation...")
        print(f"Fold sizes: {[len(fold) for fold in folds]}")
        
        # Outer loop: each fold becomes test set once
        for test_fold_idx in range(10):
            test_patients = folds[test_fold_idx]
            
            # Training patients: all other 9 folds combined
            train_patients = []
            for i in range(10):
                if i != test_fold_idx:
                    train_patients.extend(folds[i])
            
            # Inner loop: 9-fold CV on the 9 training folds
            train_folds = [folds[i] for i in range(10) if i != test_fold_idx]
            
            for val_fold_idx in range(9):
                val_patients = train_folds[val_fold_idx]
                
                # Inner training patients: remaining 8 folds
                inner_train_patients = []
                for i in range(9):
                    if i != val_fold_idx:
                        inner_train_patients.extend(train_folds[i])
                
                print(f"  Inner fold {val_fold_idx + 1}/9: Train={len(inner_train_patients)}, Val={len(val_patients)}")
                
                # Call model training function
                self.model(inner_train_patients, val_patients, test_patients)
    
    
    def left_right_split(self):
      
        return {
            'train_wrist': 'LeftWrist',
            'test_wrist': 'RightWrist'
        }
    
    def frequency_window_split(self, data, target_freq=32):
        """80:20 split with frequency downsampling (100Hz -> target_freq)"""
        if data is None:
            return None, None
        
        # Downsample from 100Hz to target frequency
        downsample_factor = target_freq / 100.0
        downsampled = resample(data, int(len(data) * downsample_factor))
        
        # Create 32-sample windows
        windows = self.create_windows(downsampled, window_size=32)
        
        if windows is None:
            return None, None
        
        # 80:20 split
        n_windows = windows.shape[0]
        split_point = int(0.85 * n_windows)
        
        train_windows = windows[:split_point]
        test_windows = windows[split_point:]
        
        return train_windows, test_windows
    
    



def main():
    # Update these paths to match your dataset location
    main_path = "pads-parkinsons-disease-smartwatch-dataset-1.0.0/movement"
    patient_path = "pads-parkinsons-disease-smartwatch-dataset-1.0.0/patients"
    
    splitter = ParkinsonsDatasetSplitter(main_path, patient_path)

if __name__ == "__main__":
    main()