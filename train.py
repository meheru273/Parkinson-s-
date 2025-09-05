import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
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
from transformers import BertTokenizer, BertModel
warnings.filterwarnings('ignore')

from dataloader import ParkinsonsDataLoader
from model import DualChannelTransformer
from metrics import calculate_metrics, save_metrics, plot_loss


def train_model(config: Dict):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Fixed: added ()
    print(f"Using device: {device}")
    
    full_dataset = ParkinsonsDataLoader(config['data_root'])

    train_dataset, val_dataset = full_dataset.get_train_test_split()
    
   
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
        model_dim=config['model_dim'],  
        num_heads=config['num_heads'],
        # Removed duplicate num_heads line
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

    # Standard CrossEntropyLoss 
    hc_pd_loss = nn.CrossEntropyLoss()
    pd_dd_loss = nn.CrossEntropyLoss()
    
    history = defaultdict(list)
    best_val_acc = 0.0
    
    # Training loop
    for epoch in range(config['num_epochs']):
        print(f"\nEpoch {epoch + 1}/{config['num_epochs']}")
        
        model.train()
        train_loss = 0.0
        hc_pd_train_pred = []
        hc_pd_train_labels = []
        pd_dd_train_pred = []
        pd_dd_train_labels = []
        
        for batch in tqdm(train_loader, desc="Training"): 
            left_sample, right_sample, hc_pd, pd_dd, patient_text = batch
            
            left_sample = left_sample.to(device)
            right_sample = right_sample.to(device)
            hc_pd = hc_pd.to(device)
            pd_dd = pd_dd.to(device)
            
            optimizer.zero_grad()

            hc_pd_logits, pd_dd_logits = model(left_sample, right_sample, patient_text, device)
            
            total_loss = 0
            loss_count = 0 
            
            # HC vs PD loss
            valid_hc_pd_mask = (hc_pd != -1)
            if valid_hc_pd_mask.any():
                valid_logits_hc = hc_pd_logits[valid_hc_pd_mask]
                valid_labels_hc = hc_pd[valid_hc_pd_mask]
                loss_hc = hc_pd_loss(valid_logits_hc, valid_labels_hc)  # Fixed: use correct loss function
                total_loss += loss_hc
                loss_count += 1

                preds_hc = torch.argmax(valid_logits_hc, dim=1)  # Fixed: dim=1, not -1
                hc_pd_train_pred.extend(preds_hc.cpu().numpy())
                hc_pd_train_labels.extend(valid_labels_hc.cpu().numpy())
            
            # PD vs DD loss  
            valid_pd_dd_mask = (pd_dd != -1)
            if valid_pd_dd_mask.any():
                valid_logits_pd = pd_dd_logits[valid_pd_dd_mask]
                valid_labels_pd = pd_dd[valid_pd_dd_mask]
                loss_pd = pd_dd_loss(valid_logits_pd, valid_labels_pd)
                total_loss += loss_pd
                loss_count += 1

                preds_pd = torch.argmax(valid_logits_pd, dim=1)
                pd_dd_train_pred.extend(preds_pd.cpu().numpy())
                pd_dd_train_labels.extend(valid_labels_pd.cpu().numpy())
                
            # Back propagation
            if loss_count > 0:
                avg_loss = total_loss / loss_count
                avg_loss.backward()
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss += avg_loss.item()
        
        train_loss /= len(train_loader)  
        
        # Calculate training metrics
        train_metrics_hc = calculate_metrics(hc_pd_train_labels, hc_pd_train_pred, "Training HC vs PD", verbose=False)
        train_metrics_pd = calculate_metrics(pd_dd_train_labels, pd_dd_train_pred, "Training PD vs DD", verbose=False)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        hc_pd_val_pred = []
        hc_pd_val_labels = []
        pd_dd_val_pred = []
        pd_dd_val_labels = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                left_sample, right_sample, hc_pd, pd_dd, patient_text = batch
                
                left_sample = left_sample.to(device)
                right_sample = right_sample.to(device)
                hc_pd = hc_pd.to(device)
                pd_dd = pd_dd.to(device)
                
                hc_pd_logits, pd_dd_logits = model(left_sample, right_sample, patient_text, device)
                
                total_loss = 0
                loss_count = 0
                
                # HC vs PD validation
                valid_hc_pd_mask = (hc_pd != -1)
                if valid_hc_pd_mask.any():
                    valid_logits_hc = hc_pd_logits[valid_hc_pd_mask]
                    valid_labels_hc = hc_pd[valid_hc_pd_mask]
                    loss_hc = hc_pd_loss(valid_logits_hc, valid_labels_hc)
                    total_loss += loss_hc
                    loss_count += 1
                    
                    preds_hc = torch.argmax(valid_logits_hc, dim=1)
                    hc_pd_val_pred.extend(preds_hc.cpu().numpy())
                    hc_pd_val_labels.extend(valid_labels_hc.cpu().numpy())
                
                # PD vs DD validation
                valid_pd_dd_mask = (pd_dd != -1)
                if valid_pd_dd_mask.any():
                    valid_logits_pd = pd_dd_logits[valid_pd_dd_mask]
                    valid_labels_pd = pd_dd[valid_pd_dd_mask]
                    loss_pd = pd_dd_loss(valid_logits_pd, valid_labels_pd)
                    total_loss += loss_pd
                    loss_count += 1
                    
                    preds_pd = torch.argmax(valid_logits_pd, dim=1)
                    pd_dd_val_pred.extend(preds_pd.cpu().numpy())
                    pd_dd_val_labels.extend(valid_labels_pd.cpu().numpy())
                
                if loss_count > 0:
                    avg_loss = total_loss / loss_count
                    val_loss += avg_loss.item()
        
        val_loss /= len(val_loader)
        
        # Calculate validation 
        print("\n" + "="*60)
        val_metrics_hc = calculate_metrics(hc_pd_val_labels, hc_pd_val_pred, "Validation HC vs PD", verbose=True)
        val_metrics_pd = calculate_metrics(pd_dd_val_labels, pd_dd_val_pred, "Validation PD vs DD", verbose=True)
        print("="*60)
        
        val_acc_hc = val_metrics_hc.get('accuracy', 0)
        val_acc_pd = val_metrics_pd.get('accuracy', 0)
        val_acc_combined = (val_acc_hc + val_acc_pd) / 2
        
        train_acc_hc = train_metrics_hc.get('accuracy', 0)
        train_acc_pd = train_metrics_pd.get('accuracy', 0)
        
        # Save metrics
        if hc_pd_val_labels:
            label_names_hc = {0: "HC", 1: "PD"}
            save_metrics(hc_pd_val_labels, hc_pd_val_pred, epoch,
                        f"metrics/epoch_{epoch}_hc_vs_pd.txt", label_names_hc)

        if pd_dd_val_labels:
            label_names_pd = {0: "PD", 1: "DD"}
            save_metrics(pd_dd_val_labels, pd_dd_val_pred, epoch,
                        f"metrics/epoch_{epoch}_pd_vs_dd.txt", label_names_pd)
        
        scheduler.step(val_loss)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc_hc'].append(train_acc_hc)
        history['train_acc_pd'].append(train_acc_pd)
        history['val_loss'].append(val_loss)
        history['val_acc_hc'].append(val_acc_hc)
        history['val_acc_pd'].append(val_acc_pd)
        history['val_acc_combined'].append(val_acc_combined)
        
        # Print epoch summary
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Train Acc - HC vs PD: {train_acc_hc:.4f}, PD vs DD: {train_acc_pd:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        print(f"Val Acc - HC vs PD: {val_acc_hc:.4f}, PD vs DD: {val_acc_pd:.4f}, Combined: {val_acc_combined:.4f}")
        
        if val_metrics_hc:
            print(f"Val F1 - HC vs PD: {val_metrics_hc.get('f1_avg', 0):.4f}")
        if val_metrics_pd:
            print(f"Val F1 - PD vs DD: {val_metrics_pd.get('f1_avg', 0):.4f}")
        
        # Save best model
        if val_acc_combined > best_val_acc:
            best_val_acc = val_acc_combined
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'val_acc_combined': val_acc_combined,
                'val_acc_hc': val_acc_hc,
                'val_acc_pd': val_acc_pd,
                'config': config
            }, 'best_model.pth')
            print("✓ New best model saved!")
    
    print(f"\nTraining completed! Best combined validation accuracy: {best_val_acc:.4f}")
    
    return {
        'best_val_accuracy': best_val_acc,
        'history': history,
        'model': model
    }


def main():
    """Main function"""
    
    config = {
        'data_root': "/kaggle/input/parkinsons/pads-parkinsons-disease-smartwatch-dataset-1.0.0",
        
        'input_dim': 6, 
        'model_dim': 64,  
        'num_heads': 8,
        'num_layers': 3,
        'd_ff': 256,
        'dropout': 0.2,  
        'seq_len': 256,
        'num_classes': 2,  
        'batch_size': 32,
        'learning_rate': 0.0005,  
        'weight_decay': 0.01,
        'num_epochs': 50,
        'patience': 15,  
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
    os.makedirs("plots", exist_ok=True)
    plot_loss(results['history'], output_path="plots")