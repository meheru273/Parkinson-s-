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

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split

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
            patient_texts=[full_dataset.patient_texts[i] for i, m in enumerate(train_mask) if m],
            patient_ids=full_dataset.patient_ids[train_mask]
        )
        
        test_dataset = type(full_dataset)(
            data_root=None,
            left_samples=full_dataset.left_samples[test_mask],
            right_samples=full_dataset.right_samples[test_mask],
            hc_vs_pd=full_dataset.hc_vs_pd[test_mask],
            pd_vs_dd=full_dataset.pd_vs_dd[test_mask],
            patient_texts=[full_dataset.patient_texts[i] for i, m in enumerate(test_mask) if m],
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


def patient_level_split_method(left_samples, right_samples, hc_vs_pd, pd_vs_dd, 
                               patient_texts, patient_ids, split_ratio=0.85):
    """
    Split data at patient level using stratified sampling to maintain class balance.
    """
    # Get unique patients and their labels
    unique_patients = np.unique(patient_ids)
    patient_labels = []
    
    for pid in unique_patients:
        # Get the label for this patient (all samples from same patient have same label)
        patient_mask = patient_ids == pid
        hc_vs_pd_label = hc_vs_pd[patient_mask][0]
        pd_vs_dd_label = pd_vs_dd[patient_mask][0]
        
        # Create a combined label for stratification
        if hc_vs_pd_label == 0:
            label = 0  # Healthy
        elif hc_vs_pd_label == 1 and pd_vs_dd_label == 0:
            label = 1  # Parkinson's
        else:
            label = 2  # Other disorders
        
        patient_labels.append(label)
    
    patient_labels = np.array(patient_labels)
    
    train_patients, test_patients = train_test_split(
        unique_patients, 
        test_size=(1 - split_ratio),
        stratify=patient_labels,
        random_state=42
    )
    
    train_patients = set(train_patients)
    test_patients = set(test_patients)
    
    train_mask = np.array([pid in train_patients for pid in patient_ids])
    test_mask = np.array([pid in test_patients for pid in patient_ids])
    
    train_data = {
        'left': left_samples[train_mask],
        'right': right_samples[train_mask],
        'hc_vs_pd': hc_vs_pd[train_mask],
        'pd_vs_dd': pd_vs_dd[train_mask],
        'texts': [patient_texts[i] for i, m in enumerate(train_mask) if m],
        'patient_ids': patient_ids[train_mask]
    }
    
    test_data = {
        'left': left_samples[test_mask],
        'right': right_samples[test_mask],
        'hc_vs_pd': hc_vs_pd[test_mask],
        'pd_vs_dd': pd_vs_dd[test_mask],
        'texts': [patient_texts[i] for i, m in enumerate(test_mask) if m],
        'patient_ids': patient_ids[test_mask]
    }
    
    # Print split info
    print(f"\nPatient-level split:")
    print(f"  Train: {len(train_patients)} patients, {len(train_data['left'])} samples")
    print(f"  Test: {len(test_patients)} patients, {len(test_data['left'])} samples")
    
    return train_data, test_data


def task_wise_split_method(left_samples, right_samples, hc_vs_pd, pd_vs_dd, 
                           patient_texts, task_names, patient_ids=None, train_tasks=None):
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
        'texts': [patient_texts[i] for i in train_indices],
        'patient_ids': patient_ids[train_indices] if patient_ids is not None else None
    }
    
    test_data = {
        'left': left_samples[test_indices],
        'right': right_samples[test_indices],
        'hc_vs_pd': hc_vs_pd[test_indices],
        'pd_vs_dd': pd_vs_dd[test_indices],
        'texts': [patient_texts[i] for i in test_indices],
        'patient_ids': patient_ids[test_indices] if patient_ids is not None else None
    }
    
    return train_data, test_data


####################dataloader###################
class ParkinsonsDataLoader(Dataset):
    
    def __init__(self, data_root: str = None, window_size: int = 256, 
                 left_samples=None, right_samples=None, 
                 hc_vs_pd=None, pd_vs_dd=None, patient_texts=None,
                 apply_dowsampling=True,
                 apply_bandpass_filter=True, apply_prepare_text=True, **kwargs):
        
        self.left_samples = []
        self.right_samples = []
        self.hc_vs_pd = []
        self.pd_vs_dd = []
        self.sample_splits = []  # DEPRECATED: no longer used for splitting
        self.patient_texts = []
        self.patient_ids = []  
        self.task_names = []   
        self.apply_dowsampling = apply_dowsampling
        self.apply_bandpass_filter = apply_bandpass_filter
        self.apply_prepare_text = apply_prepare_text
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
            if patient_texts is not None:
                self.patient_texts = list(patient_texts) if not isinstance(patient_texts, list) else patient_texts
        
            self.patient_ids = kwargs.get('patient_ids', [])
            if self.patient_ids is not None and len(self.patient_ids) > 0:
                self.patient_ids = np.array(self.patient_ids) if not isinstance(self.patient_ids, np.ndarray) else self.patient_ids


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
                else:
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
        if split_type == 1:
            # Patient-level split with stratification
            split_ratio = kwargs.get('split_ratio', 0.85)
            
            if len(self.patient_ids) == 0:
                raise ValueError("Patient IDs are required for patient-level split")
            
            train_data, test_data = patient_level_split_method(
                self.left_samples, self.right_samples, 
                self.hc_vs_pd, self.pd_vs_dd, self.patient_texts,
                self.patient_ids, split_ratio
            )
            
            train_dataset = ParkinsonsDataLoader(
                data_root=None,
                left_samples=train_data['left'],
                right_samples=train_data['right'],
                hc_vs_pd=train_data['hc_vs_pd'],
                pd_vs_dd=train_data['pd_vs_dd'],
                patient_texts=train_data['texts'],
                patient_ids=train_data['patient_ids']
            )
            
            test_dataset = ParkinsonsDataLoader(
                data_root=None,
                left_samples=test_data['left'],
                right_samples=test_data['right'],
                hc_vs_pd=test_data['hc_vs_pd'],
                pd_vs_dd=test_data['pd_vs_dd'],
                patient_texts=test_data['texts'],
                patient_ids=test_data['patient_ids']
            )
            
            return train_dataset, test_dataset
            
        elif split_type == 2:
            # Task-based split
            train_tasks = kwargs.get('train_tasks', self.tasks[:8])
            train_data, test_data = task_wise_split_method(
                self.left_samples, self.right_samples,
                self.hc_vs_pd, self.pd_vs_dd, self.patient_texts,
                self.task_names, self.patient_ids, train_tasks
            )
            
            train_dataset = ParkinsonsDataLoader(
                data_root=None,
                left_samples=train_data['left'],
                right_samples=train_data['right'],
                hc_vs_pd=train_data['hc_vs_pd'],
                pd_vs_dd=train_data['pd_vs_dd'],
                patient_texts=train_data['texts'],
                patient_ids=train_data['patient_ids']
            )
            
            test_dataset = ParkinsonsDataLoader(
                data_root=None,
                left_samples=test_data['left'],
                right_samples=test_data['right'],
                hc_vs_pd=test_data['hc_vs_pd'],
                pd_vs_dd=test_data['pd_vs_dd'],
                patient_texts=test_data['texts'],
                patient_ids=test_data['patient_ids']
            )
            
            return train_dataset, test_dataset
            
        elif split_type == 3:
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
        patient_text = self.patient_texts[idx]
        
        return left_sample, right_sample, hc_vs_pd.squeeze(), pd_vs_dd.squeeze(), patient_text