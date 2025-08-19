import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import defaultdict
import logging
from typing import Dict, List, Tuple
import warnings
import torch.nn.functional as F
import math

import os
import numpy as np
import json
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

warnings.filterwarnings('ignore')

from dataloader import ParkinsonsDataLoader
from model import DualChannelTransformer

class EarlyStopping:
    """Early stopping utility"""
    
    def __init__(self, patience=7, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        
    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience

def save_metrics(y_true, y_pred, out_path="metrics.txt", label_names=None, append=False):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    if y_true.size == 0:
        raise ValueError("y_true is empty")

    labels = np.unique(np.concatenate([y_true, y_pred]))
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, sup = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    # prepare simple text
    lines = []
    lines.append(f"Accuracy: {acc:.4f}")
    lines.append("")
    lines.append("Per-class (label, support, precision, recall, f1):")
    for i, lab in enumerate(labels):
        name = label_names.get(int(lab), str(lab)) if label_names else str(lab)
        lines.append(f"{name}\t{int(sup[i])}\t{prec[i]:.4f}\t{rec[i]:.4f}\t{f1[i]:.4f}")
    lines.append("")
    lines.append("Confusion matrix (rows=true, cols=pred):")
    # header
    header = "\t" + "\t".join([label_names.get(int(l), str(l)) if label_names else str(l) for l in labels])
    lines.append(header)
    for i, row in enumerate(cm):
        row_label = label_names.get(int(labels[i]), str(labels[i])) if label_names else str(labels[i])
        lines.append(row_label + "\t" + "\t".join(str(int(x)) for x in row))

    # write file
    mode = "a" if append else "w"
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, mode) as f:
        f.write("\n".join(lines) + "\n")

    # also save JSON (same name .json) for programmatic use
    json_out = out_path.rstrip(".txt") + ".json"
    data = {
        "accuracy": float(acc),
        "per_class": {int(l): {"support": int(sup[i]), "precision": float(prec[i]), 
                               "recall": float(rec[i]), "f1": float(f1[i])} for i, l in enumerate(labels)},
        "confusion_matrix": cm.tolist(),
        "labels": labels.tolist()
    }
    with open(json_out, "w") as fj:
        json.dump(data, fj, indent=2)

    return out_path


def train_model(config: Dict):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")


    full_dataset = ParkinsonsDataLoader(config['data_root'])


    # Get train/test split based on per-patient 80:20 split
    train_dataset, val_dataset = full_dataset.get_train_test_split()
    
    print(f"Total samples: {len(full_dataset)}")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Test samples: {len(val_dataset)}")
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True,
        num_workers=config['num_workers']
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False,
        num_workers=config['num_workers']
    )
    
    model = DualChannelTransformer(
        input_dim=config['input_dim'],
        d_model=config['d_model'],
        num_heads=config['num_heads'],
        num_layers=config['num_layers'],
        d_ff=config['d_ff'],
        dropout=config['dropout'],
        seq_len=config['seq_len'],
        num_classes=config['num_classes']
    ).to(device)
    
    
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    criterion_hc_vs_pd = nn.CrossEntropyLoss()
    criterion_pd_vs_dd = nn.CrossEntropyLoss()
    
    early_stopping = EarlyStopping(patience=config['patience'])
    
    # Training history
    history = defaultdict(list)
    best_val_accuracy = 0.0
    
    # Training loop
    for epoch in range(config['num_epochs']):
        print(f"Epoch {epoch+1}/{config['num_epochs']}")
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_preds_hc_vs_pd = []
        train_labels_hc_vs_pd = []
        train_preds_pd_vs_dd = []
        train_labels_pd_vs_dd = []
        
        for batch in tqdm(train_loader, desc="Training"):
            left_sample, right_sample, hc_vs_pd_left, pd_vs_dd_left, hc_vs_pd_right, pd_vs_dd_right = batch
            
            left_sample = left_sample.to(device)
            right_sample = right_sample.to(device)
            hc_vs_pd_left = hc_vs_pd_left.to(device)
            pd_vs_dd_left = pd_vs_dd_left.to(device)
            
            optimizer.zero_grad()
            
            logits_hc_vs_pd, logits_pd_vs_dd = model(left_sample, right_sample)
            
            total_loss = 0
            loss_count = 0
            
            # HC vs PD loss (for samples where hc_vs_pd_left != -1)
            valid_hc_vs_pd_mask = (hc_vs_pd_left != -1)
            if valid_hc_vs_pd_mask.any():
                valid_logits_hc = logits_hc_vs_pd[valid_hc_vs_pd_mask]
                valid_labels_hc = hc_vs_pd_left[valid_hc_vs_pd_mask]
                loss_hc = criterion_hc_vs_pd(valid_logits_hc, valid_labels_hc)
                total_loss += loss_hc
                loss_count += 1
                
            
                preds_hc = torch.argmax(valid_logits_hc, dim=1)
                train_preds_hc_vs_pd.extend(preds_hc.cpu().numpy())
                train_labels_hc_vs_pd.extend(valid_labels_hc.cpu().numpy())
            
            # PD vs DD loss 
            valid_pd_vs_dd_mask = (pd_vs_dd_left != -1)
            if valid_pd_vs_dd_mask.any():
                valid_logits_pd = logits_pd_vs_dd[valid_pd_vs_dd_mask]
                valid_labels_pd = pd_vs_dd_left[valid_pd_vs_dd_mask]
                loss_pd = criterion_pd_vs_dd(valid_logits_pd, valid_labels_pd)
                total_loss += loss_pd
                loss_count += 1
                
               
                preds_pd = torch.argmax(valid_logits_pd, dim=1)
                train_preds_pd_vs_dd.extend(preds_pd.cpu().numpy())
                train_labels_pd_vs_dd.extend(valid_labels_pd.cpu().numpy())
            
            if loss_count > 0:
                avg_loss = total_loss / loss_count
                avg_loss.backward()
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_loss += avg_loss.item()
        
        train_loss /= len(train_loader)
        
        # Calculate training accuracies
        train_acc_hc = accuracy_score(train_labels_hc_vs_pd, train_preds_hc_vs_pd) if train_labels_hc_vs_pd else 0
        train_acc_pd = accuracy_score(train_labels_pd_vs_dd, train_preds_pd_vs_dd) if train_labels_pd_vs_dd else 0
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_preds_hc_vs_pd = []
        val_labels_hc_vs_pd = []
        val_preds_pd_vs_dd = []
        val_labels_pd_vs_dd = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                left_sample, right_sample, hc_vs_pd_left, pd_vs_dd_left, hc_vs_pd_right, pd_vs_dd_right = batch
                
                # Move to device
                left_sample = left_sample.to(device)
                right_sample = right_sample.to(device)
                hc_vs_pd_left = hc_vs_pd_left.to(device)
                pd_vs_dd_left = pd_vs_dd_left.to(device)
                
               
                logits_hc_vs_pd, logits_pd_vs_dd = model(left_sample, right_sample)
                
                # Calculate losses only for valid labels
                total_loss = 0
                loss_count = 0
                
                # HC vs PD validation
                valid_hc_vs_pd_mask = (hc_vs_pd_left != -1)
                if valid_hc_vs_pd_mask.any():
                    valid_logits_hc = logits_hc_vs_pd[valid_hc_vs_pd_mask]
                    valid_labels_hc = hc_vs_pd_left[valid_hc_vs_pd_mask]
                    loss_hc = criterion_hc_vs_pd(valid_logits_hc, valid_labels_hc)
                    total_loss += loss_hc
                    loss_count += 1
                    
                    preds_hc = torch.argmax(valid_logits_hc, dim=1)
                    val_preds_hc_vs_pd.extend(preds_hc.cpu().numpy())
                    val_labels_hc_vs_pd.extend(valid_labels_hc.cpu().numpy())
                
                # PD vs DD validation
                valid_pd_vs_dd_mask = (pd_vs_dd_left != -1)
                if valid_pd_vs_dd_mask.any():
                    valid_logits_pd = logits_pd_vs_dd[valid_pd_vs_dd_mask]
                    valid_labels_pd = pd_vs_dd_left[valid_pd_vs_dd_mask]
                    loss_pd = criterion_pd_vs_dd(valid_logits_pd, valid_labels_pd)
                    total_loss += loss_pd
                    loss_count += 1
                    
                    preds_pd = torch.argmax(valid_logits_pd, dim=1)
                    val_preds_pd_vs_dd.extend(preds_pd.cpu().numpy())
                    val_labels_pd_vs_dd.extend(valid_labels_pd.cpu().numpy())
                
                if loss_count > 0:
                    avg_loss = total_loss / loss_count
                    val_loss += avg_loss.item()
        
        val_loss /= len(val_loader)
        
        # Calculate validation accuracies
        val_acc_hc = accuracy_score(val_labels_hc_vs_pd, val_preds_hc_vs_pd) if val_labels_hc_vs_pd else 0
        val_acc_pd = accuracy_score(val_labels_pd_vs_dd, val_preds_pd_vs_dd) if val_labels_pd_vs_dd else 0
        
        # Combined accuracy (average of both tasks)
        val_acc_combined = (val_acc_hc + val_acc_pd) / 2
        
        # Calculate detailed metrics and save
        if val_labels_hc_vs_pd:
            label_names_hc = {0: "HC", 1: "PD"}
            save_metrics(val_labels_hc_vs_pd, val_preds_hc_vs_pd, 
                        f"metrics/epoch_{epoch}_hc_vs_pd.txt", label_names_hc)
        
        if val_labels_pd_vs_dd:
            label_names_pd = {0: "PD", 1: "DD"}
            save_metrics(val_labels_pd_vs_dd, val_preds_pd_vs_dd, 
                        f"metrics/epoch_{epoch}_pd_vs_dd.txt", label_names_pd)
        
        # Update scheduler
        scheduler.step(val_loss)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc_hc'].append(train_acc_hc)
        history['train_acc_pd'].append(train_acc_pd)
        history['val_loss'].append(val_loss)
        history['val_acc_hc'].append(val_acc_hc)
        history['val_acc_pd'].append(val_acc_pd)
        history['val_acc_combined'].append(val_acc_combined)
        
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Train Acc - HC vs PD: {train_acc_hc:.4f}, PD vs DD: {train_acc_pd:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        print(f"Val Acc - HC vs PD: {val_acc_hc:.4f}, PD vs DD: {val_acc_pd:.4f}, Combined: {val_acc_combined:.4f}")
        
        # Save best model
        if val_acc_combined > best_val_accuracy:
            best_val_accuracy = val_acc_combined
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'val_acc_combined': val_acc_combined,
                'val_acc_hc': val_acc_hc,
                'val_acc_pd': val_acc_pd,
                'config': config
            }, 'best_model.pth')
        
        # Early stopping check
        if early_stopping(val_loss):
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    print(f"Training completed! Best combined validation accuracy: {best_val_accuracy:.4f}")
    
    return {
        'best_val_accuracy': best_val_accuracy,
        'history': history,
        'model': model
    }


def main():
    """Main function"""
    
    # Configuration
    config = {

        'data_root': "/kaggle/input/parkinsons/pads-parkinsons-disease-smartwatch-dataset-1.0.0",
        
        'input_dim': 6, 
        'd_model': 64,
        'num_heads': 8,
        'num_layers': 3,
        'd_ff': 256,
        'dropout': 0.1,
        'seq_len': 32,
        'num_classes': 2,  
        
        # Training parameters
        'batch_size': 32,
        'learning_rate': 0.001,
        'weight_decay': 0.01,
        'num_epochs': 50,
        'patience': 10,
        'num_workers': 0,
    }
    
    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    results = train_model(config)
    
    if results is None:
        print("Training failed due to data loading issues.")
        return None
    
    return results


if __name__ == "__main__":
    results = main()