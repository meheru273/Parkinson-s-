import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pathlib
import json
import os
import csv
from tqdm import tqdm
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GATConv, global_mean_pool
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_curve, auc
from sklearn.manifold import TSNE
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
import torch.optim as optim


###############Helper functions (from your code)##########
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


###############Patient-Level Graph DataLoader##########
class PatientGraphDataLoader(Dataset):
    def __init__(self, data_root: str = None, window_size: int = 256,
                 patient_data=None,
                 apply_downsampling=True,
                 apply_bandpass_filter=True,
                 use_complete_graph=False,
                 edge_threshold=0.3):
        
        self.window_size = window_size
        self.apply_downsampling = apply_downsampling
        self.apply_bandpass_filter = apply_bandpass_filter
        self.use_complete_graph = use_complete_graph
        self.edge_threshold = edge_threshold
        self.data_root = data_root
        
        self.tasks = ["CrossArms", "DrinkGlas", "Entrainment", "HoldWeight", "LiftHold", 
                     "PointFinger", "Relaxed", "StretchHold", "TouchIndex", "TouchNose"]
        
        self.patient_data = {}
        
        if data_root is not None:
            self.patients_template = pathlib.Path(data_root) / "patients" / "patient_{p:03d}.json"
            self.timeseries_template = pathlib.Path(data_root) / "movement" / "timeseries" / "{N:03d}_{X}_{Y}.txt"
            self._load_patient_data()
        elif patient_data is not None:
            self.patient_data = patient_data
        
        self.graphs = []
        self._create_graphs()
    
    
    def _load_patient_data(self):
        """Load all patient data organized by patient and task."""
        patient_ids_list = list(range(1, 470))
        
        print(f"Loading patient-level data for {len(patient_ids_list)} patients...")
        
        for patient_id in tqdm(patient_ids_list, desc="Loading patients"):
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
                
                patient_left_data = {task: [] for task in self.tasks}
                patient_right_data = {task: [] for task in self.tasks}
                
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
                        
                        if self.apply_downsampling:
                            left_data = downsample(left_data)
                            right_data = downsample(right_data)
                        
                        if self.apply_bandpass_filter:
                            left_data = bandpass_filter(left_data)
                            right_data = bandpass_filter(right_data)
                        
                        left_windows = create_windows(left_data, self.window_size, overlap=overlap)
                        right_windows = create_windows(right_data, self.window_size, overlap=overlap)
                        
                        if left_windows is not None:
                            patient_left_data[task] = left_windows
                        if right_windows is not None:
                            patient_right_data[task] = right_windows
                    
                    except Exception as e:
                        continue
                
                if any(len(v) > 0 for v in patient_left_data.values()):
                    self.patient_data[patient_id] = {
                        'left': patient_left_data,
                        'right': patient_right_data,
                        'labels': (hc_vs_pd_label, pd_vs_dd_label)
                    }
            
            except Exception as e:
                print(f"Error loading patient {patient_id}: {e}")
                continue
        
        print(f"Loaded {len(self.patient_data)} patients with valid data")
    
    
    def _aggregate_windows_for_node(self, windows):
        if len(windows) == 0:
            return None
        
        flattened_windows = windows.reshape(len(windows), -1)
        aggregated = np.mean(flattened_windows, axis=0)
        
        return aggregated
    
    
    def _create_task_graph(self, task_data, hc_vs_pd_label, pd_vs_dd_label, wrist_type, patient_id):
        """Create a graph for one wrist of one patient."""
        node_features = []
        valid_tasks = []
        
        for task in self.tasks:
            windows = task_data.get(task, [])
            if len(windows) > 0:
                features = self._aggregate_windows_for_node(windows)
                if features is not None:
                    node_features.append(features)
                    valid_tasks.append(task)
        
        if len(node_features) == 0:
            return None
        
        node_features = np.array(node_features)
        x = torch.FloatTensor(node_features)
        n_nodes = len(node_features)
        
        if self.use_complete_graph:
            edge_index = []
            edge_attr = []
            for i in range(n_nodes):
                for j in range(n_nodes):
                    if i != j:
                        edge_index.append([i, j])
                        edge_attr.append([1.0])
            
            edge_index = torch.LongTensor(edge_index).t().contiguous()
            edge_attr = torch.FloatTensor(edge_attr)
        
        else:
            similarity_matrix = cosine_similarity(node_features)
            edge_index = []
            edge_attr = []
            
            for i in range(n_nodes):
                for j in range(n_nodes):
                    if i != j and similarity_matrix[i, j] > self.edge_threshold:
                        edge_index.append([i, j])
                        edge_attr.append([similarity_matrix[i, j]])
            
            if len(edge_index) == 0:
                edge_index = [[i, i] for i in range(n_nodes)]
                edge_attr = [[1.0] for _ in range(n_nodes)]
            
            edge_index = torch.LongTensor(edge_index).t().contiguous()
            edge_attr = torch.FloatTensor(edge_attr)
        
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y_hc_pd=torch.LongTensor([hc_vs_pd_label]),
            y_pd_dd=torch.LongTensor([pd_vs_dd_label]),
            patient_id=patient_id,
            wrist_type=wrist_type,
            num_nodes=n_nodes
        )
        
        return data
    
    
    def _create_graphs(self):
        """Create graph objects for all patients."""
        print(f"Creating graphs for {len(self.patient_data)} patients...")
        
        for patient_id, data in tqdm(self.patient_data.items(), desc="Creating graphs"):
            left_data = data['left']
            right_data = data['right']
            hc_vs_pd_label, pd_vs_dd_label = data['labels']
            
            left_graph = self._create_task_graph(
                left_data, hc_vs_pd_label, pd_vs_dd_label, 'left', patient_id
            )
            
            right_graph = self._create_task_graph(
                right_data, hc_vs_pd_label, pd_vs_dd_label, 'right', patient_id
            )
            
            if left_graph is not None and right_graph is not None:
                self.graphs.append({
                    'left': left_graph,
                    'right': right_graph,
                    'patient_id': patient_id,
                    'y_hc_pd': hc_vs_pd_label,
                    'y_pd_dd': pd_vs_dd_label
                })
        
        print(f"Created {len(self.graphs)} patient-level graph pairs")
    
    
    def get_train_test_split(self, k=5):
        """K-fold patient-level split using stratification."""
        if self.data_root is None:
            raise ValueError("data_root is required for K-fold split")
        
        patient_ids = list(self.patient_data.keys())
        patient_labels = []
        
        for pid in patient_ids:
            hc_vs_pd, pd_vs_dd = self.patient_data[pid]['labels']
            if hc_vs_pd == 0:
                label = 0
            elif hc_vs_pd == 1 and pd_vs_dd == 0:
                label = 1
            else:
                label = 2
            patient_labels.append(label)
        
        print(f"Total patients: {len(patient_ids)} (HC={patient_labels.count(0)}, PD={patient_labels.count(1)}, DD={patient_labels.count(2)})")
        
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
        fold_datasets = []
        
        for fold_id, (train_idx, test_idx) in enumerate(skf.split(patient_ids, patient_labels)):
            train_patients = set([patient_ids[i] for i in train_idx])
            test_patients = set([patient_ids[i] for i in test_idx])
            
            train_patient_data = {pid: self.patient_data[pid] for pid in train_patients}
            test_patient_data = {pid: self.patient_data[pid] for pid in test_patients}
            
            train_dataset = PatientGraphDataLoader(
                data_root=None,
                window_size=self.window_size,
                patient_data=train_patient_data,
                use_complete_graph=self.use_complete_graph,
                edge_threshold=self.edge_threshold
            )
            
            test_dataset = PatientGraphDataLoader(
                data_root=None,
                window_size=self.window_size,
                patient_data=test_patient_data,
                use_complete_graph=self.use_complete_graph,
                edge_threshold=self.edge_threshold
            )
            
            train_labels = [train_patient_data[pid]['labels'] for pid in train_patients]
            test_labels = [test_patient_data[pid]['labels'] for pid in test_patients]
            
            train_hc = sum(1 for hc, pd in train_labels if hc == 0)
            train_pd = sum(1 for hc, pd in train_labels if hc == 1 and pd == 0)
            train_dd = sum(1 for hc, pd in train_labels if pd == 1)
            
            test_hc = sum(1 for hc, pd in test_labels if hc == 0)
            test_pd = sum(1 for hc, pd in test_labels if hc == 1 and pd == 0)
            test_dd = sum(1 for hc, pd in test_labels if pd == 1)
            
            print(f"\nFold {fold_id+1}/{k}:")
            print(f"  Train: {len(train_dataset)} patients (HC={train_hc}, PD={train_pd}, DD={train_dd})")
            print(f"  Test:  {len(test_dataset)} patients (HC={test_hc}, PD={test_pd}, DD={test_dd})")
            
            fold_datasets.append((train_dataset, test_dataset))
        
        return fold_datasets
    
    
    def __len__(self):
        return len(self.graphs)
    
    
    def __getitem__(self, idx):
        return self.graphs[idx]


###############Custom Collate Function##########
def collate_patient_graphs(batch):
    """Custom collate to handle patient-level graph pairs."""
    left_graphs = [item['left'] for item in batch]
    right_graphs = [item['right'] for item in batch]
    
    left_batch = Batch.from_data_list(left_graphs)
    right_batch = Batch.from_data_list(right_graphs)
    
    y_hc_pd = torch.LongTensor([item['y_hc_pd'] for item in batch])
    y_pd_dd = torch.LongTensor([item['y_pd_dd'] for item in batch])
    
    return {
        'left': left_batch,
        'right': right_batch,
        'y_hc_pd': y_hc_pd,
        'y_pd_dd': y_pd_dd
    }


###############GAT Model##########
class PatientGATModel(nn.Module):
    """GAT-based model for patient-level classification."""
    
    def __init__(self, input_dim, hidden_dim=64, num_heads=4, dropout=0.3, 
                 num_classes_hc_pd=2, num_classes_pd_dd=2):
        super().__init__()
        
        self.gat1 = GATConv(input_dim, hidden_dim, heads=num_heads, dropout=dropout)
        self.gat2 = GATConv(hidden_dim * num_heads, hidden_dim, heads=num_heads, dropout=dropout)
        self.gat3 = GATConv(hidden_dim * num_heads, hidden_dim, heads=1, concat=False, dropout=dropout)
        
        self.dropout = nn.Dropout(dropout)
        self.pool = global_mean_pool
        
        # Classification heads matching your base model structure
        self.fc_hc_pd = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes_hc_pd)
        )
        
        self.fc_pd_dd = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes_pd_dd)
        )
    
    
    def forward_graph(self, data):
        """Process a single graph (left or right)."""
        x, edge_index = data.x, data.edge_index
        
        x = self.gat1(x, edge_index)
        x = F.elu(x)
        x = self.dropout(x)
        
        x = self.gat2(x, edge_index)
        x = F.elu(x)
        x = self.dropout(x)
        
        x = self.gat3(x, edge_index)
        x = F.elu(x)
        
        x = self.pool(x, data.batch)
        
        return x
    
    
    def forward(self, batch):
        """Forward pass - matches your base model interface."""
        left_embedding = self.forward_graph(batch['left'])
        right_embedding = self.forward_graph(batch['right'])
        
        combined_embedding = torch.cat([left_embedding, right_embedding], dim=1)
        
        logits_hc_pd = self.fc_hc_pd(combined_embedding)
        logits_pd_dd = self.fc_pd_dd(combined_embedding)
        
        return logits_hc_pd, logits_pd_dd
    
    
    def get_features(self, batch):
        """Extract features for t-SNE - matches your base model interface."""
        left_embedding = self.forward_graph(batch['left'])
        right_embedding = self.forward_graph(batch['right'])
        combined_embedding = torch.cat([left_embedding, right_embedding], dim=1)
        return combined_embedding


###############Evaluation Functions (from your code)##########
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
                     fold_metrics_hc, fold_metrics_pd):
    os.makedirs("metrics", exist_ok=True)

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
        hc_filename = f"metrics/hc_vs_pd_metrics{fold_suffix}.csv"
        write_csv(hc_filename, fold_metrics_hc)
        print(f"✓ HC vs PD metrics saved: {hc_filename}")

    if fold_metrics_pd:
        pd_filename = f"metrics/pd_vs_dd_metrics{fold_suffix}.csv"
        write_csv(pd_filename, fold_metrics_pd)
        print(f"✓ PD vs DD metrics saved: {pd_filename}")


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
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(features)-1), n_iter=1000)
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


###############Training Functions (adapted for graph model)##########
def train_single_epoch(model, dataloader, criterion_hc, criterion_pd, optimizer, device):
    model.train()
    train_loss = 0.0
    hc_pd_train_pred, hc_pd_train_labels = [], []
    pd_dd_train_pred, pd_dd_train_labels = [], []
    
    for batch in tqdm(dataloader, desc="Training"):
        # Move graph batches to device
        batch['left'] = batch['left'].to(device)
        batch['right'] = batch['right'].to(device)
        hc_pd = batch['y_hc_pd'].to(device)
        pd_dd = batch['y_pd_dd'].to(device)
        
        optimizer.zero_grad()
        
        # CRITICAL DIFFERENCE: No device parameter needed, batch contains graphs
        hc_pd_logits, pd_dd_logits = model(batch)
        
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += avg_loss.item()
    
    train_loss /= len(dataloader)
    
    # Calculate training metrics using your function
    train_metrics_hc = calculate_metrics(hc_pd_train_labels, hc_pd_train_pred, 
                                        "Training HC vs PD", verbose=False)
    train_metrics_pd = calculate_metrics(pd_dd_train_labels, pd_dd_train_pred, 
                                        "Training PD vs DD", verbose=False)
    
    return train_loss, train_metrics_hc, train_metrics_pd


def validate_single_epoch(model, dataloader, criterion_hc, criterion_pd, device):
    """Validate for one epoch - ADAPTED FOR GRAPH MODEL"""
    model.eval()
    val_loss = 0.0
    hc_pd_val_pred, hc_pd_val_labels, hc_pd_val_probs = [], [], []
    pd_dd_val_pred, pd_dd_val_labels, pd_dd_val_probs = [], [], []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            batch['left'] = batch['left'].to(device)
            batch['right'] = batch['right'].to(device)
            hc_pd = batch['y_hc_pd'].to(device)
            pd_dd = batch['y_pd_dd'].to(device)
            
            hc_pd_logits, pd_dd_logits = model(batch)
            
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
    """Extract features for t-SNE - ADAPTED FOR GRAPH MODEL"""
    model.eval()
    all_features = []
    all_hc_pd_labels = []
    all_pd_dd_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting features"):
            batch['left'] = batch['left'].to(device)
            batch['right'] = batch['right'].to(device)
            
            # CRITICAL: Graph model has get_features method that takes batch dict
            features = model.get_features(batch)
            
            all_features.append(features.cpu().numpy())
            all_hc_pd_labels.append(batch['y_hc_pd'].numpy())
            all_pd_dd_labels.append(batch['y_pd_dd'].numpy())
    
    all_features = np.vstack(all_features)
    all_hc_pd_labels = np.concatenate(all_hc_pd_labels)
    all_pd_dd_labels = np.concatenate(all_pd_dd_labels)
    
    return all_features, all_hc_pd_labels, all_pd_dd_labels


def train_graph_model(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    os.makedirs("metrics", exist_ok=True)
    
    full_dataset = PatientGraphDataLoader(
        data_root=config['data_root'],
        window_size=config.get('window_size', 256),
        apply_downsampling=config.get('apply_downsampling', True),
        apply_bandpass_filter=config.get('apply_bandpass_filter', True),
        use_complete_graph=config.get('use_complete_graph', False),
        edge_threshold=config.get('edge_threshold', 0.3)
    )
    
    # K-fold split
    fold_datasets = full_dataset.get_train_test_split(k=config['num_folds'])
    num_folds = len(fold_datasets)
    
    all_fold_results = []
    
    for fold_idx in range(num_folds):
        print(f"\n{'='*60}")
        print(f"Starting Fold {fold_idx+1}/{num_folds}")
        print(f"{'='*60}")
        
        train_dataset, val_dataset = fold_datasets[fold_idx]
        
        # DIFFERENCE 2: Use custom collate function
        train_loader = DataLoader(
            train_dataset, 
            batch_size=config['batch_size'], 
            shuffle=True,
            collate_fn=collate_patient_graphs,
            num_workers=config.get('num_workers', 0)
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=config['batch_size'], 
            shuffle=False,
            collate_fn=collate_patient_graphs,
            num_workers=config.get('num_workers', 0)
        )
        input_dim = train_dataset[0]['left'].x.shape[1]
        
        model = PatientGATModel(
            input_dim=input_dim,
            hidden_dim=config.get('hidden_dim', 64),
            num_heads=config.get('num_heads', 4),
            dropout=config.get('dropout', 0.3),
            num_classes_hc_pd=2,
            num_classes_pd_dd=2
        ).to(device)
        
        optimizer = optim.AdamW(model.parameters(), 
                               lr=config['learning_rate'], 
                               weight_decay=config['weight_decay'])
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        criterion_hc = torch.nn.CrossEntropyLoss()
        criterion_pd = torch.nn.CrossEntropyLoss()
        
        history = defaultdict(list)
        best_val_acc = 0.0
        best_epoch = 0
        
        fold_metrics_hc = []
        fold_metrics_pd = []
        
        best_hc_pd_probs = None
        best_hc_pd_preds = None
        best_hc_pd_labels = None
        best_pd_dd_probs = None
        best_pd_dd_preds = None
        best_pd_dd_labels = None
        
        for epoch in range(config['num_epochs']):
            print(f"\nEpoch {epoch + 1}/{config['num_epochs']}")
            
            # Training phase
            train_loss, train_metrics_hc, train_metrics_pd = train_single_epoch(
                model, train_loader, criterion_hc, criterion_pd, optimizer, device
            )
            
            # Validation phase
            val_results = validate_single_epoch(
                model, val_loader, criterion_hc, criterion_pd, device
            )
            val_loss, hc_pd_val_pred, hc_pd_val_labels, hc_pd_val_probs, \
            pd_dd_val_pred, pd_dd_val_labels, pd_dd_val_probs = val_results
            
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
            
            # Store metrics
            if hc_pd_val_labels:
                fold_metrics_hc.append({
                    'epoch': epoch + 1,
                    'predictions': hc_pd_val_pred.copy(),
                    'labels': hc_pd_val_labels.copy(),
                    'metrics': val_metrics_hc
                })
            
            if pd_dd_val_labels:
                fold_metrics_pd.append({
                    'epoch': epoch + 1,
                    'predictions': pd_dd_val_pred.copy(),
                    'labels': pd_dd_val_labels.copy(),
                    'metrics': val_metrics_pd
                })
            
            val_acc_hc = val_metrics_hc.get('accuracy', 0)
            val_acc_pd = val_metrics_pd.get('accuracy', 0)
            val_acc_combined = (val_acc_hc + val_acc_pd) / 2
            
            train_acc_hc = train_metrics_hc.get('accuracy', 0)
            train_acc_pd = train_metrics_pd.get('accuracy', 0)
            
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
            
            # Save best model
            if val_acc_combined > best_val_acc:
                best_val_acc = val_acc_combined
                best_epoch = epoch + 1
                
                if hc_pd_val_probs:
                    best_hc_pd_probs = np.array(hc_pd_val_probs)
                    best_hc_pd_preds = np.array(hc_pd_val_pred)
                    best_hc_pd_labels = np.array(hc_pd_val_labels)
                
                if pd_dd_val_probs:
                    best_pd_dd_probs = np.array(pd_dd_val_probs)
                    best_pd_dd_preds = np.array(pd_dd_val_pred)
                    best_pd_dd_labels = np.array(pd_dd_val_labels)
                
                model_save_name = f'best_graph_model_fold_{fold_idx+1}.pth'
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'fold': fold_idx,
                    'epoch': epoch,
                    'val_acc_combined': val_acc_combined,
                    'val_acc_hc': val_acc_hc,
                    'val_acc_pd': val_acc_pd,
                    'config': config
                }, model_save_name)
                print(f"✓ New best model saved: {model_save_name}")
        
        # Save metrics
        if config.get('save_metrics', True):
            fold_suffix = f"_fold_{fold_idx+1}"
            if fold_metrics_hc and fold_metrics_pd:
                save_fold_metric(fold_idx, fold_suffix, best_epoch, best_val_acc,
                               fold_metrics_hc, fold_metrics_pd)
        
        # Extract features
        fold_features, fold_hc_pd_labels, fold_pd_dd_labels = extract_features(
            model, val_loader, device
        )
        
        fold_result = {
            'best_val_accuracy': best_val_acc,
            'history': history,
            'features': fold_features,
            'hc_pd_labels': fold_hc_pd_labels,
            'pd_dd_labels': fold_pd_dd_labels
        }
        all_fold_results.append(fold_result)
        
        # Create plots
        if config.get('create_plots', True):
            plot_dir = f"plots/fold_{fold_idx+1}"
            os.makedirs(plot_dir, exist_ok=True)
            
            plot_loss(history['train_loss'], history['val_loss'], 
                     f"{plot_dir}/loss.png")
            
            if best_hc_pd_probs is not None and len(best_hc_pd_labels) > 0:
                plot_roc_curves(best_hc_pd_labels, best_hc_pd_preds, best_hc_pd_probs,
                              f"{plot_dir}/roc_hc_vs_pd.png")
            
            if best_pd_dd_probs is not None and len(best_pd_dd_labels) > 0:
                plot_roc_curves(best_pd_dd_labels, best_pd_dd_preds, best_pd_dd_probs,
                              f"{plot_dir}/roc_pd_vs_dd.png")
            
            if fold_features is not None:
                plot_tsne(fold_features, fold_hc_pd_labels, fold_pd_dd_labels, 
                         output_dir=plot_dir)
    
    # Print final summary
    print("\n" + "="*60)
    print("TRAINING COMPLETE - SUMMARY")
    print("="*60)
    for i, result in enumerate(all_fold_results):
        print(f"Fold {i+1}: Best Val Accuracy = {result['best_val_accuracy']:.4f}")
    avg_acc = np.mean([r['best_val_accuracy'] for r in all_fold_results])
    print(f"\nAverage Accuracy Across Folds: {avg_acc:.4f}")
    print("="*60)
    
    return all_fold_results


def main():
    """Main function with graph model configuration"""
    
    config = {
        # Data parameters
        'data_root': "/kaggle/input/parkinsons/pads-parkinsons-disease-smartwatch-dataset-1.0.0",
        'window_size': 256,
        'apply_downsampling': True,
        'apply_bandpass_filter': True,
        'num_folds': 5,
        
        # Graph parameters
        'use_complete_graph': False,  
        'edge_threshold': 0.3,
        
        # Model parameters
        'hidden_dim': 64,
        'num_heads': 4,
        'dropout': 0.3,
        
        # Training parameters
        'batch_size': 32,  
        'learning_rate': 0.001,
        'weight_decay': 0.01,
        'num_epochs': 100,
        'num_workers': 0,
        
        # Flags
        'save_metrics': True,
        'create_plots': True,
    }
    
    results = train_graph_model(config)
    
    return results


if __name__ == "__main__":
    results = main()