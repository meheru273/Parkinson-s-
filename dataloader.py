import pathlib
import numpy as np
import json
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from typing import Dict, List, Tuple
import warnings


def create_windows(data, window_size=256):
    
    n_samples, n_channels = data.shape
    n_windows = n_samples // window_size

    trimmed_data = data[:n_windows * window_size]
    windows = trimmed_data.reshape(n_windows, window_size, n_channels)
        
    return windows

def create_overlapping_window(data, window_size, overlap):
    n_samples, n_channels = data.shape
    stride = window_size - overlap
    windows = []
    for start in range(0, n_samples - window_size + 1, stride):
        end = start + window_size
        windows.append(data[start:end])
    return np.array(windows)

def downsample(data, original_freq=100, target_freq=64):
    
    step = int(original_freq // target_freq)  
    if step > 1:
        return data[::step, :]
    return data


class ParkinsonsDataLoader(Dataset):
    
    def __init__(self, data_root: str = None, window_size: int = 256, 
                 left_samples=None, right_samples=None, 
                 hc_vs_pd_left=None, pd_vs_dd_left=None,
                 hc_vs_pd_right=None, pd_vs_dd_right=None):
        
        
        self.left_samples = []
        self.right_samples = []
        self.hc_vs_pd_left = []
        self.hc_vs_pd_right = []
        self.pd_vs_dd_left = []
        self.pd_vs_dd_right = []
        self.sample_splits = []
        
        if data_root is not None:
            
            self.data_root = data_root
            self.window_size = window_size
            self.patients_template = pathlib.Path(data_root) / "patients" / "patient_{p:03d}.json"
            self.timeseries_template = pathlib.Path(data_root) / "movement" / "timeseries" / "{N:03d}_{X}_{Y}.txt"
            
            # Tasks 
            self.tasks = ["CrossArms", "DrinkGlas", "Entrainment", "HoldWeight", "LiftHold", 
                         "PointFinger", "Relaxed", "StretchHold", "TouchIndex", "TouchNose"]
            self.wrists = ["LeftWrist", "RightWrist"]
            
            self.patient_ids = list(range(1, 470))
            print(f"Dataset: {len(self.patient_ids)} patients (001-469)")
        
            self._load_data()
        else:
            if left_samples is not None:
                self.left_samples = np.array(left_samples) if not isinstance(left_samples, np.ndarray) else left_samples
            if right_samples is not None:
                self.right_samples = np.array(right_samples) if not isinstance(right_samples, np.ndarray) else right_samples
            if hc_vs_pd_left is not None:
                self.hc_vs_pd_left = np.array(hc_vs_pd_left) if not isinstance(hc_vs_pd_left, np.ndarray) else hc_vs_pd_left
            if pd_vs_dd_left is not None:
                self.pd_vs_dd_left = np.array(pd_vs_dd_left) if not isinstance(pd_vs_dd_left, np.ndarray) else pd_vs_dd_left
            if hc_vs_pd_right is not None:
                self.hc_vs_pd_right = np.array(hc_vs_pd_right) if not isinstance(hc_vs_pd_right, np.ndarray) else hc_vs_pd_right
            if pd_vs_dd_right is not None:
                self.pd_vs_dd_right = np.array(pd_vs_dd_right) if not isinstance(pd_vs_dd_right, np.ndarray) else pd_vs_dd_right
    
    def _load_data(self):
        for patient_id in tqdm(self.patient_ids, desc="Loading patients"):
            patient_path = pathlib.Path(str(self.patients_template).format(p=patient_id))
            
            if not patient_path.exists():
                print(f"no path found for patient {patient_id}")
                continue
                
            try:
                with open(patient_path, 'r') as f:
                    metadata = json.load(f)
                
                condition = metadata.get('condition', 'Healthy')
                
                if condition == 'Healthy':
                    hc_vs_pd_label = 0  # Healthy
                    pd_vs_dd_label = -1  # Not applicable for PD vs DD task
                elif 'Parkinson' in condition:
                    hc_vs_pd_label = 1  # Parkinson's
                    pd_vs_dd_label = 0   # Parkinson's (for PD vs DD)
                else:  
                    hc_vs_pd_label = -1  # Not applicable for HC vs PD task
                    pd_vs_dd_label = 1   # Other disorders
                
                
                patient_left_samples = []
                patient_right_samples = []
                patient_samples = 0
                
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
                            left_data = left_data[:, :6]  #six channels
                        if left_data.shape[0] > 50:
                            left_data = left_data[50:, :] # 0.5 sec
                        
                        if right_data.shape[1] > 6:
                            right_data = right_data[:, :6]
                        if right_data.shape[0] > 50:
                            right_data = right_data[50:, :]
                        
                        # Downsample 
                        left_data = downsample(left_data)
                        right_data = downsample(right_data)
                        
                        if left_data is None or right_data is None:
                            continue
                            
                        left_window = create_windows(left_data, self.window_size)
                        right_window = create_windows(right_data, self.window_size)
                        
                        if left_window is not None and right_window is not None:
                        
                            for i in range(len(left_window)):
                                patient_left_samples.append(left_window[i])
                            for i in range(len(right_window)):
                                patient_right_samples.append(right_window[i])
                            
                            patient_samples += 1
                                        
                    except Exception as e:
                        print(f"Error loading data for patient {patient_id}, task {task}: {e}")
                        continue
                
                if len(patient_left_samples) > 0:
                    n_patient_samples = len(patient_left_samples) #both samples have same patients
                    split_point = int(0.85 * n_patient_samples)
                    

                    for i in range(n_patient_samples):
                        self.left_samples.append(patient_left_samples[i])
                        self.right_samples.append(patient_right_samples[i])
                        self.hc_vs_pd_left.append(hc_vs_pd_label)
                        self.hc_vs_pd_right.append(hc_vs_pd_label)
                        self.pd_vs_dd_left.append(pd_vs_dd_label)
                        self.pd_vs_dd_right.append(pd_vs_dd_label)
                        
                        
                        if i < split_point:
                            self.sample_splits.append(0)  # train
                        else:
                            self.sample_splits.append(1)  # test
                    
                    print(f"Patient {patient_id}: condition='{condition}', HC_vs_PD={hc_vs_pd_label}, PD_vs_DD={pd_vs_dd_label}, total_samples={n_patient_samples}, train={split_point}, test={n_patient_samples-split_point}")
                    
            except Exception as e:
                print(f"Error loading patient {patient_id}: {e}")
                continue
        
        self.left_samples = np.array(self.left_samples)
        self.right_samples = np.array(self.right_samples)
        self.hc_vs_pd_right = np.array(self.hc_vs_pd_right)
        self.pd_vs_dd_right = np.array(self.pd_vs_dd_right)
        self.hc_vs_pd_left = np.array(self.hc_vs_pd_left)
        self.pd_vs_dd_left = np.array(self.pd_vs_dd_left)
        self.sample_splits = np.array(self.sample_splits)

        print(f"Dataset loaded: {len(self.left_samples)} paired samples")
        if len(self.left_samples) > 0:
        
            hc_count = np.sum(self.hc_vs_pd_left == 0)
            pd_count = np.sum(self.hc_vs_pd_left == 1)
            pd_vs_dd_pd_count = np.sum(self.pd_vs_dd_left == 0)
            pd_vs_dd_dd_count = np.sum(self.pd_vs_dd_left == 1)
            
           
            train_count = np.sum(self.sample_splits == 0)
            test_count = np.sum(self.sample_splits == 1)
            
            print(f"HC vs PD - HC: {hc_count}, PD: {pd_count}")
            print(f"PD vs DD - PD: {pd_vs_dd_pd_count}, DD: {pd_vs_dd_dd_count}")
            print(f"Split - Train: {train_count} ({train_count/len(self.sample_splits)*100:.1f}%), Test: {test_count} ({test_count/len(self.sample_splits)*100:.1f}%)")
    
    def get_train_test_split(self):
        train_mask = self.sample_splits == 0
        test_mask = self.sample_splits == 1
        
        train_dataset = ParkinsonsDataLoader(
            data_root=None,
            left_samples=self.left_samples[train_mask],
            right_samples=self.right_samples[train_mask],
            hc_vs_pd_left=self.hc_vs_pd_left[train_mask],
            pd_vs_dd_left=self.pd_vs_dd_left[train_mask],
            hc_vs_pd_right=self.hc_vs_pd_right[train_mask],
            pd_vs_dd_right=self.pd_vs_dd_right[train_mask]
        )
        
        test_dataset = ParkinsonsDataLoader(
            data_root=None,
            left_samples=self.left_samples[test_mask],
            right_samples=self.right_samples[test_mask],
            hc_vs_pd_left=self.hc_vs_pd_left[test_mask],
            pd_vs_dd_left=self.pd_vs_dd_left[test_mask],
            hc_vs_pd_right=self.hc_vs_pd_right[test_mask],
            pd_vs_dd_right=self.pd_vs_dd_right[test_mask]
        )
        
        return train_dataset, test_dataset
    
    def __len__(self):
        return len(self.left_samples) if hasattr(self, 'left_samples') and isinstance(self.left_samples, (list, np.ndarray)) else 0
    
    def __getitem__(self, idx):
        left_sample = torch.FloatTensor(self.left_samples[idx])
        right_sample = torch.FloatTensor(self.right_samples[idx])
        hc_vs_pd_left = torch.LongTensor([self.hc_vs_pd_left[idx]])
        pd_vs_dd_left = torch.LongTensor([self.pd_vs_dd_left[idx]])
        hc_vs_pd_right = torch.LongTensor([self.hc_vs_pd_right[idx]])
        pd_vs_dd_right = torch.LongTensor([self.pd_vs_dd_right[idx]])

        return left_sample, right_sample, hc_vs_pd_left.squeeze(), pd_vs_dd_left.squeeze(), hc_vs_pd_right.squeeze(), pd_vs_dd_right.squeeze()