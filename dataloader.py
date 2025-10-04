import pathlib
import numpy as np
import json
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from typing import Dict, List, Tuple
import warnings
from scipy.signal import butter, filtfilt
warnings.filterwarnings("ignore", category=UserWarning)

###############Helper functions##########
def create_windows(data, window_size=256, overlap=0):
    """Create sliding windows from time series data"""
    n_samples, n_channels = data.shape
    
    step = int(window_size * (1 - overlap))   #step size
    
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


#prepare text 
def prepare_text(metadata, questionnaires):
    text_array = []
    
    if metadata:
        text_array.append(f"Age: {metadata.get('age', 'unknown')}")
        text_array.append(f"Gender: {metadata.get('gender', 'unknown')}")
        if metadata.get('age_at_diagnosis'):
            text_array.append(f"Age at diagnosis: {metadata.get('age_at_diagnosis')}")
        if metadata.get('disease_comment'):
            text_array.append(f"Clinical notes: {metadata.get('disease_comment')}")
    
    if questionnaires and 'item' in questionnaires:
        for item in questionnaires['item']:
            q_text = item.get('text', '')
            q_answer = item.get('answer', '')
            if q_text and q_answer:
                text_array.append(f"Q: {q_text} A: {q_answer}")
    
    return " ".join(text_array) if text_array else "No information available."


###############splitting methods################
def k_fold_split_method(left_patient_samples, right_patient_samples, patient_labels_hc, 
                        patient_labels_pd, patient_texts, patient_ids, k=10):
    
    unique_patients = sorted(list(set(patient_ids)))
    n_patients = len(unique_patients)
    
    np.random.seed(42)
    np.random.shuffle(unique_patients)
    
    fold_size = n_patients // k
    remainder = n_patients % k
    
    patient_folds = []
    start = 0
    for i in range(k):
        size = fold_size + (1 if i < remainder else 0)
        patient_folds.append(unique_patients[start:start + size])
        start += size
        
    all_folds = []
    
    for fold_idx in range(k):
        test_patients = set(patient_folds[fold_idx])
        
        train_indices = []
        test_indices = []
        
        for idx, pid in enumerate(patient_ids):
            if pid in test_patients:
                test_indices.append(idx)
            else:
                train_indices.append(idx)


        train_data = {
            'left': left_patient_samples[train_indices],
            'right': right_patient_samples[train_indices],
            'hc_vs_pd': patient_labels_hc[train_indices],
            'pd_vs_dd': patient_labels_pd[train_indices],
            'texts': [patient_texts[i] for i in train_indices]
        }
        
        test_data = {
            'left': left_patient_samples[test_indices],
            'right': right_patient_samples[test_indices],
            'hc_vs_pd': patient_labels_hc[test_indices],
            'pd_vs_dd': patient_labels_pd[test_indices],
            'texts': [patient_texts[i] for i in test_indices]
        }
        
        all_folds.append((train_data, test_data))
        
        print(f"Fold {fold_idx + 1}/{k}: Train samples: {len(train_indices)}, Test samples: {len(test_indices)}")
    
    return all_folds


def split_signal_method(left_samples, right_samples, hc_vs_pd, pd_vs_dd, patient_texts, split_ratio=0.85):
    """Current method - split by signal index"""
    n_samples = len(left_samples)
    split_point = int(split_ratio * n_samples)
    
    train_data = {
        'left': left_samples[:split_point],
        'right': right_samples[:split_point],
        'hc_vs_pd': hc_vs_pd[:split_point],
        'pd_vs_dd': pd_vs_dd[:split_point],
        'texts': patient_texts[:split_point]
    }
    
    test_data = {
        'left': left_samples[split_point:],
        'right': right_samples[split_point:],
        'hc_vs_pd': hc_vs_pd[split_point:],
        'pd_vs_dd': pd_vs_dd[split_point:],
        'texts': patient_texts[split_point:]
    }
    
    return train_data, test_data


def task_wise_split_method(left_samples, right_samples, hc_vs_pd, pd_vs_dd, 
                           patient_texts, task_names, train_tasks):
    """Split by tasks - use certain tasks for training, others for testing"""
    train_indices = []
    test_indices = []
    
    for idx, task in enumerate(task_names):
        if task in train_tasks:
            train_indices.append(idx)
        else:
            test_indices.append(idx)
    
    train_data = {
        'left': left_samples[train_indices],
        'right': right_samples[train_indices],
        'hc_vs_pd': hc_vs_pd[train_indices],
        'pd_vs_dd': pd_vs_dd[train_indices],
        'texts': [patient_texts[i] for i in train_indices]
    }
    
    test_data = {
        'left': left_samples[test_indices],
        'right': right_samples[test_indices],
        'hc_vs_pd': hc_vs_pd[test_indices],
        'pd_vs_dd': pd_vs_dd[test_indices],
        'texts': [patient_texts[i] for i in test_indices]
    }
    
    return train_data, test_data


####################dataloader###################
class ParkinsonsDataLoader(Dataset):
    
    def __init__(self, data_root: str = None, window_size: int = 256, 
                 left_samples=None, right_samples=None, 
                 hc_vs_pd=None, pd_vs_dd=None, patient_texts=None,
                 apply_dowsampling = True,
                 apply_bandpass_filter = True, apply_prepare_text=True):
        
        self.left_samples = []
        self.right_samples = []
        self.hc_vs_pd = []
        self.pd_vs_dd = []
        self.sample_splits = []
        self.patient_texts = []
        self.patient_ids = []  
        self.task_names = []   
        self.apply_dowsampling = apply_dowsampling
        self.apply_bandpass_filter = apply_bandpass_filter
        self.apply_prepare_text = apply_prepare_text


        if data_root is not None:
            self.data_root = data_root
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
            if patient_texts is not None:
                self.patient_texts = list(patient_texts) if not isinstance(patient_texts, list) else patient_texts


    def _load_data(self):
        for patient_id in tqdm(self.patient_ids_list, desc="Loading patients"):
            patient_path = pathlib.Path(str(self.patients_template).format(p=patient_id))
            questionnaire_path = pathlib.Path(str(self.questionnaires_template).format(p=patient_id))
            
            if not patient_path.exists():
                continue
                
            try:
                with open(patient_path, 'r') as f:
                    metadata = json.load(f)
                
                condition = metadata.get('condition', '')
                questionnaire = {}
                try:
                    with open(questionnaire_path, 'r') as f:
                        questionnaire = json.load(f)
                except:
                    pass


                if self.apply_prepare_text:
                    per_patient_text = prepare_text(metadata, questionnaire)
                else :
                    per_patient_text = ""


                if condition == 'Healthy':
                    hc_vs_pd_label = 0  # Healthy
                    pd_vs_dd_label = -1  # Not applicable for PD vs DD 
                    overlap = 0.70
                elif 'Parkinson' in condition:
                    hc_vs_pd_label = 1  
                    pd_vs_dd_label = 0   # Parkinson's for PD vs DD
                    overlap = 0
                else:  
                    hc_vs_pd_label = -1  # Not applicable for HC vs PD 
                    pd_vs_dd_label = 1   # Other disorders
                    overlap = 0.65


                patient_left_samples = []
                patient_right_samples = []
                patient_sample_texts = []
                patient_task_names = []
                
                for task in self.tasks:
                    #insert task name 
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
                                patient_sample_texts.append(per_patient_text)
                                patient_task_names.append(task)
                        
                    except Exception as e:
                        print(f"Error loading data for patient {patient_id}, task {task}: {e}")
                        continue
                
                # Add all samples from this patient
                if len(patient_left_samples) > 0:
                    n_samples = len(patient_left_samples)
                    
                    for i in range(n_samples):
                        self.left_samples.append(patient_left_samples[i])
                        self.right_samples.append(patient_right_samples[i])
                        self.patient_texts.append(patient_sample_texts[i])
                        self.hc_vs_pd.append(hc_vs_pd_label)
                        self.pd_vs_dd.append(pd_vs_dd_label)
                        self.patient_ids.append(patient_id)
                        self.task_names.append(patient_task_names[i])
                        
                        # Simple split for backwards compatibility
                        if i < int(0.85 * n_samples):
                            self.sample_splits.append(0)  # train
                        else:
                            self.sample_splits.append(1)  # test
                
            except Exception as e:
                print(f"Error loading patient {patient_id}: {e}")
                continue
        
        self.left_samples = np.array(self.left_samples)
        self.right_samples = np.array(self.right_samples)
        self.hc_vs_pd = np.array(self.hc_vs_pd)
        self.pd_vs_dd = np.array(self.pd_vs_dd)
        self.sample_splits = np.array(self.sample_splits)
        self.patient_ids = np.array(self.patient_ids)
        self.task_names = np.array(self.task_names)


    def get_train_test_split(self, split_type=1, **kwargs):
        if split_type == 1:
            # Signal-based split
            split_ratio = kwargs.get('split_ratio', 0.85)
            train_data, test_data = split_signal_method(
                self.left_samples, self.right_samples, 
                self.hc_vs_pd, self.pd_vs_dd, self.patient_texts, 
                split_ratio
            )
            
            train_dataset = ParkinsonsDataLoader(
                data_root=None,
                left_samples=train_data['left'],
                right_samples=train_data['right'],
                hc_vs_pd=train_data['hc_vs_pd'],
                pd_vs_dd=train_data['pd_vs_dd'],
                patient_texts=train_data['texts']
            )
            
            test_dataset = ParkinsonsDataLoader(
                data_root=None,
                left_samples=test_data['left'],
                right_samples=test_data['right'],
                hc_vs_pd=test_data['hc_vs_pd'],
                pd_vs_dd=test_data['pd_vs_dd'],
                patient_texts=test_data['texts']
            )
            
            return train_dataset, test_dataset
            
        elif split_type == 2:
            # Task-based split
            train_tasks = kwargs.get('train_tasks', self.tasks[:8])
            train_data, test_data = task_wise_split_method(
                self.left_samples, self.right_samples,
                self.hc_vs_pd, self.pd_vs_dd, self.patient_texts,
                self.task_names, train_tasks
            )
            
            train_dataset = ParkinsonsDataLoader(
                data_root=None,
                left_samples=train_data['left'],
                right_samples=train_data['right'],
                hc_vs_pd=train_data['hc_vs_pd'],
                pd_vs_dd=train_data['pd_vs_dd'],
                patient_texts=train_data['texts'],
            )
            
            test_dataset = ParkinsonsDataLoader(
                data_root=None,
                left_samples=test_data['left'],
                right_samples=test_data['right'],
                hc_vs_pd=test_data['hc_vs_pd'],
                pd_vs_dd=test_data['pd_vs_dd'],
                patient_texts=test_data['texts']
            )
            
            return train_dataset, test_dataset
        #k-fold split    
        elif split_type == 3:
            k = kwargs.get('k', 10)
            all_folds = k_fold_split_method(
                self.left_samples, self.right_samples,
                self.hc_vs_pd, self.pd_vs_dd,
                self.patient_texts, self.patient_ids, k
            )
            
            fold_datasets = []
            for train_data, test_data in all_folds:
                train_dataset = ParkinsonsDataLoader(
                    data_root=None,
                    left_samples=train_data['left'],
                    right_samples=train_data['right'],
                    hc_vs_pd=train_data['hc_vs_pd'],
                    pd_vs_dd=train_data['pd_vs_dd'],
                    patient_texts=train_data['texts']
                )
                
                test_dataset = ParkinsonsDataLoader(
                    data_root=None,
                    left_samples=test_data['left'],
                    right_samples=test_data['right'],
                    hc_vs_pd=test_data['hc_vs_pd'],
                    pd_vs_dd=test_data['pd_vs_dd'],
                    patient_texts=test_data['texts']
                )
                
                fold_datasets.append((train_dataset, test_dataset))
            
            return fold_datasets
        
        else:
            train_mask = self.sample_splits == 0
            test_mask = self.sample_splits == 1
            
            train_texts = [self.patient_texts[i] for i in range(len(self.patient_texts)) if train_mask[i]]
            test_texts = [self.patient_texts[i] for i in range(len(self.patient_texts)) if test_mask[i]]
            
            train_dataset = ParkinsonsDataLoader(
                data_root=None,
                left_samples=self.left_samples[train_mask],
                right_samples=self.right_samples[train_mask],
                hc_vs_pd=self.hc_vs_pd[train_mask],
                pd_vs_dd=self.pd_vs_dd[train_mask],
                patient_texts=train_texts
            )
            
            test_dataset = ParkinsonsDataLoader(
                data_root=None,
                left_samples=self.left_samples[test_mask],
                right_samples=self.right_samples[test_mask],
                hc_vs_pd=self.hc_vs_pd[test_mask],
                pd_vs_dd=self.pd_vs_dd[test_mask],
                patient_texts=test_texts
            )
            
            return train_dataset, test_dataset


    def __len__(self):
        return len(self.left_samples) if hasattr(self, 'left_samples') and isinstance(self.left_samples, (list, np.ndarray)) else 0
    
    def __getitem__(self, idx):
        left_sample = torch.FloatTensor(self.left_samples[idx])
        right_sample = torch.FloatTensor(self.right_samples[idx])
        hc_vs_pd = torch.LongTensor([self.hc_vs_pd[idx]])
        pd_vs_dd = torch.LongTensor([self.pd_vs_dd[idx]])
        patient_text = self.patient_texts[idx]
        
        return left_sample, right_sample, hc_vs_pd.squeeze(), pd_vs_dd.squeeze(), patient_text
