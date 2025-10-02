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
warnings.filterwarnings('ignore')

from dataloader import ParkinsonsDataLoader
from model import DualChannelTransformer
from metrics import calculate_metrics, save_metrics, plot_loss
from tnse import plot_tsne

def train_model(config: Dict):
   
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    os.makedirs("metrics", exist_ok=True)
    
    
    full_dataset = ParkinsonsDataLoader(config['data_root'],
        apply_dowsampling=config['apply_downsampling'],
        apply_bandpass_filter=config['apply_bandpass_filter'],
        apply_prepare_text=config.get('apply_prepare_text', False)
    )

    split_type = config.get('split_type', 3)
    
    if split_type == 3:  
        fold_datasets = full_dataset.get_train_test_split(split_type=3, k=config['num_folds'])
        num_folds = len(fold_datasets)
    else:  
        train_dataset, val_dataset = full_dataset.get_train_test_split(
            split_type=split_type, 
            split_ratio=config.get('split_ratio', 0.85),
            train_tasks=config.get('train_tasks', None)
        )
        fold_datasets = [(train_dataset, val_dataset)]
        num_folds = 1
    
    all_fold_results = []
    
    # Training loop for each fold
    for fold_idx in range(num_folds):
        print(f"\n{'='*70}")
        if num_folds > 1:
            print(f"Starting Fold {fold_idx+1}/{num_folds}")
            train_dataset, val_dataset = fold_datasets[fold_idx]
            
        train_loader = DataLoader(
            train_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=config['num_workers']
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=config['num_workers']
        )

        # Initialize model
        model = DualChannelTransformer(
            input_dim=config['input_dim'],
            model_dim=config['model_dim'],  
            num_heads=config['num_heads'],
            num_layers=config['num_layers'],
            d_ff=config['d_ff'],
            dropout=config['dropout'],
            seq_len=config['seq_len'],
            num_classes=config['num_classes'],
            use_text=config.get('use_text', False)
        ).to(device)

        # Initialize optimizer and scheduler
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )

        # Loss functions
        hc_pd_loss = nn.CrossEntropyLoss()
        pd_dd_loss = nn.CrossEntropyLoss()
        
        # Training history
        history = defaultdict(list)
        best_val_acc = 0.0
        best_epoch = 0
        fold_features = None
        fold_hc_pd_labels = None
        fold_pd_dd_labels = None
        
        # Collect all epoch metrics for this fold
        fold_metrics_hc = []
        fold_metrics_pd = []
        
        # Training epochs
        for epoch in range(config['num_epochs']):
            print(f"\nEpoch {epoch + 1}/{config['num_epochs']}")
            
            # ---------------Training phase--------------
            model.train()
            train_loss = 0.0
            hc_pd_train_pred, hc_pd_train_labels = [], []
            pd_dd_train_pred, pd_dd_train_labels = [], []
            
            for batch in tqdm(train_loader, desc="Training"): 
                left_sample, right_sample, hc_pd, pd_dd, patient_text = batch
                
                left_sample = left_sample.to(device)
                right_sample = right_sample.to(device)
                hc_pd = hc_pd.to(device)
                pd_dd = pd_dd.to(device)
                
                optimizer.zero_grad()

                # Forward pass
                text_input = patient_text if config.get('use_text', False) else None
                hc_pd_logits, pd_dd_logits = model(left_sample, right_sample, text_input, device)
                
                total_loss = 0
                loss_count = 0 
                
                # HC vs PD loss
                valid_hc_pd_mask = (hc_pd != -1)
                if valid_hc_pd_mask.any():
                    valid_logits_hc = hc_pd_logits[valid_hc_pd_mask]
                    valid_labels_hc = hc_pd[valid_hc_pd_mask]
                    loss_hc = hc_pd_loss(valid_logits_hc, valid_labels_hc)
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
                    loss_pd = pd_dd_loss(valid_logits_pd, valid_labels_pd)
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
            
            train_loss /= len(train_loader)  
            
            # Calculate training metrics
            train_metrics_hc = calculate_metrics(hc_pd_train_labels, hc_pd_train_pred, "Training HC vs PD", verbose=False)
            train_metrics_pd = calculate_metrics(pd_dd_train_labels, pd_dd_train_pred, "Training PD vs DD", verbose=False)
            
            # --------------Validation phase-------------
            model.eval()
            val_loss = 0.0
            hc_pd_val_pred, hc_pd_val_labels = [], []
            pd_dd_val_pred, pd_dd_val_labels = [], []
            
            # Collect features for t-SNE (last epoch only)
            epoch_features = []
            epoch_hc_pd_labels = []
            epoch_pd_dd_labels = []
            
            with torch.no_grad():
                for batch in tqdm(val_loader, desc="Validation"):
                    left_sample, right_sample, hc_pd, pd_dd, patient_text = batch
                    
                    left_sample = left_sample.to(device)
                    right_sample = right_sample.to(device)
                    hc_pd = hc_pd.to(device)
                    pd_dd = pd_dd.to(device)
                    
                    text_input = patient_text if config.get('use_text', False) else None
                    hc_pd_logits, pd_dd_logits = model(left_sample, right_sample, text_input, device)
                    
                    # Extract features for t-SNE (last epoch)
                    if epoch == config['num_epochs'] - 1:
                        left_encoded = model.left_projection(left_sample)
                        right_encoded = model.right_projection(right_sample)
                        left_encoded = model.positional_encoding(left_encoded)
                        right_encoded = model.positional_encoding(right_encoded)
                        
                        for layer in model.layers:
                            left_encoded, right_encoded = layer(left_encoded, right_encoded)
                        
                        left_pool = model.global_pool(left_encoded.transpose(1, 2)).squeeze(-1)
                        right_pool = model.global_pool(right_encoded.transpose(1, 2)).squeeze(-1)
                        features = torch.cat([left_pool, right_pool], dim=1)
                        
                        epoch_features.append(features.cpu().numpy())
                        epoch_hc_pd_labels.extend(hc_pd.cpu().numpy())
                        epoch_pd_dd_labels.extend(pd_dd.cpu().numpy())
                    
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
            
            # Store features from last epoch
            if epoch == config['num_epochs'] - 1 and epoch_features:
                fold_features = np.concatenate(epoch_features, axis=0)
                fold_hc_pd_labels = np.array(epoch_hc_pd_labels)
                fold_pd_dd_labels = np.array(epoch_pd_dd_labels)
            
            # Calculate validation metrics
            print("\n" + "="*60)
            val_metrics_hc = calculate_metrics(hc_pd_val_labels, hc_pd_val_pred, 
                                             f"{'Fold ' + str(fold_idx+1) + ' ' if num_folds > 1 else ''}Validation HC vs PD", 
                                             verbose=True)
            val_metrics_pd = calculate_metrics(pd_dd_val_labels, pd_dd_val_pred, 
                                             f"{'Fold ' + str(fold_idx+1) + ' ' if num_folds > 1 else ''}Validation PD vs DD", 
                                             verbose=True)
            print("="*60)
            
            # Store metrics for this epoch
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
            
            # Calculate combined accuracy
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
            
            # Print epoch summary
            print(f"\n{'Fold ' + str(fold_idx+1) + ', ' if num_folds > 1 else ''}Epoch {epoch+1} Summary:")
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Train Acc - HC vs PD: {train_acc_hc:.4f}, PD vs DD: {train_acc_pd:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            print(f"Val Acc - HC vs PD: {val_acc_hc:.4f}, PD vs DD: {val_acc_pd:.4f}, Combined: {val_acc_combined:.4f}")
            
            # Save best model
            if val_acc_combined > best_val_acc:
                best_val_acc = val_acc_combined
                best_epoch = epoch + 1
                model_save_name = f'best_model{"_fold_" + str(fold_idx+1) if num_folds > 1 else ""}.pth'
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'fold': fold_idx if num_folds > 1 else None,
                    'epoch': epoch,
                    'val_acc_combined': val_acc_combined,
                    'val_acc_hc': val_acc_hc,
                    'val_acc_pd': val_acc_pd,
                    'config': config
                }, model_save_name)
                print(f"✓ New best model saved: {model_save_name}")
        
        # Save all metrics for this fold to single files
        if config.get('save_metrics', True):
            fold_suffix = f"_fold_{fold_idx+1}" if num_folds > 1 else ""
            
            # Save HC vs PD metrics
            if fold_metrics_hc:
                hc_filename = f"metrics/hc_vs_pd_metrics{fold_suffix}.txt"
                with open(hc_filename, 'w') as f:
                    f.write(f"{'='*70}\n")
                    f.write(f"{'FOLD ' + str(fold_idx+1) + ' ' if num_folds > 1 else ''}HC vs PD METRICS - ALL EPOCHS\n")
                    f.write(f"Best Epoch: {best_epoch} (Combined Accuracy: {best_val_acc:.4f})\n")
                    f.write(f"{'='*70}\n\n")
                    
                    for epoch_data in fold_metrics_hc:
                        f.write(f"EPOCH {epoch_data['epoch']}:\n")
                        f.write(f"Accuracy: {epoch_data['metrics'].get('accuracy', 0):.4f}\n")
                        f.write(f"Precision: {epoch_data['metrics'].get('precision', 0):.4f}\n")
                        f.write(f"Recall: {epoch_data['metrics'].get('recall', 0):.4f}\n")
                        f.write(f"F1-Score: {epoch_data['metrics'].get('f1', 0):.4f}\n")
                        
                        # Add confusion matrix if available
                        if len(epoch_data['labels']) > 0:
                            cm = confusion_matrix(epoch_data['labels'], epoch_data['predictions'])
                            f.write(f"Confusion Matrix:\n{cm}\n")
                        f.write("-" * 50 + "\n\n")
                
                print(f"✓ HC vs PD metrics saved: {hc_filename}")
            
            # Save PD vs DD metrics
            if fold_metrics_pd:
                pd_filename = f"metrics/pd_vs_dd_metrics{fold_suffix}.txt"
                with open(pd_filename, 'w') as f:
                    f.write(f"{'='*70}\n")
                    f.write(f"{'FOLD ' + str(fold_idx+1) + ' ' if num_folds > 1 else ''}PD vs DD METRICS - ALL EPOCHS\n")
                    f.write(f"Best Epoch: {best_epoch} (Combined Accuracy: {best_val_acc:.4f})\n")
                    f.write(f"{'='*70}\n\n")
                    
                    for epoch_data in fold_metrics_pd:
                        f.write(f"EPOCH {epoch_data['epoch']}:\n")
                        f.write(f"Accuracy: {epoch_data['metrics'].get('accuracy', 0):.4f}\n")
                        f.write(f"Precision: {epoch_data['metrics'].get('precision', 0):.4f}\n")
                        f.write(f"Recall: {epoch_data['metrics'].get('recall', 0):.4f}\n")
                        f.write(f"F1-Score: {epoch_data['metrics'].get('f1', 0):.4f}\n")
                        
                        # Add confusion matrix if available
                        if len(epoch_data['labels']) > 0:
                            cm = confusion_matrix(epoch_data['labels'], epoch_data['predictions'])
                            f.write(f"Confusion Matrix:\n{cm}\n")
                        f.write("-" * 50 + "\n\n")
                
                print(f"✓ PD vs DD metrics saved: {pd_filename}")
        
        # Store fold results
        fold_result = {
            'best_val_accuracy': best_val_acc,
            'history': history,
            'features': fold_features,
            'hc_pd_labels': fold_hc_pd_labels,
            'pd_dd_labels': fold_pd_dd_labels
        }
        all_fold_results.append(fold_result)
        
        # Create plots if enabled
        if config.get('create_plots', True):
            plot_dir = f"plots/{'fold_' + str(fold_idx+1) if num_folds > 1 else 'single_run'}"
            os.makedirs(plot_dir, exist_ok=True)
            
            plot_loss(history, f"{plot_dir}/loss.png")
            if fold_features is not None:
                plot_tsne(fold_features, fold_hc_pd_labels, fold_pd_dd_labels, output_dir=plot_dir)
    
    return all_fold_results

def main():
    """Main function with configurable parameters"""
    
    config = {
        'data_root': "/kaggle/input/parkinsons/pads-parkinsons-disease-smartwatch-dataset-1.0.0",
        'apply_downsampling': True,
        'apply_bandpass_filter': True,
        'apply_prepare_text': False,  
        'split_type': 3, 
        'split_ratio': 0.85,  
        'train_tasks': None,
        'num_folds': 5,
        
        'input_dim': 6, 
        'model_dim': 64,  
        'num_heads': 8,
        'num_layers': 3,
        'd_ff': 256,
        'dropout': 0.2,  
        'seq_len': 256,
        'num_classes': 2,
        'use_text': False, 

        'batch_size': 32,
        'learning_rate': 0.0005,  
        'weight_decay': 0.01,
        'num_epochs': 30,
        'num_workers': 0,

        'save_metrics': True,
        'create_plots': True
    }
    results = train_model(config)
    
    if results is None:
        print("Training failed due to data loading issues.")
        return None
    
    return results

if __name__ == "__main__":
    results = main()