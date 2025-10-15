import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix, roc_curve, auc
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
warnings.filterwarnings('ignore')
from dataloader import ParkinsonsDataLoader
from model import DualChannelTransformer
from metrics import plot_loss, plot_roc_curves, plot_tsne, save_fold_metric, calculate_metrics


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
            shuffle=True,  
            num_workers=config['num_workers']
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=config['num_workers']
        )

        # model
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

        # optimizer and scheduler
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )

        hc_pd_loss = nn.CrossEntropyLoss()
        pd_dd_loss = nn.CrossEntropyLoss()
        
        history = defaultdict(list)
        best_val_acc = 0.0
        best_epoch = 0
        fold_features = None
        fold_hc_pd_labels = None
        fold_pd_dd_labels = None
        
        fold_metrics_hc = []
        fold_metrics_pd = []
        
        # Store best predictions and probabilities for ROC
        best_hc_pd_probs = None
        best_hc_pd_preds = None
        best_hc_pd_labels = None
        best_pd_dd_probs = None
        best_pd_dd_preds = None
        best_pd_dd_labels = None
        
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
            hc_pd_val_pred, hc_pd_val_labels, hc_pd_val_probs = [], [], []
            pd_dd_val_pred, pd_dd_val_labels, pd_dd_val_probs = [], [], []
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
                        probs_hc = F.softmax(valid_logits_hc, dim=1)[:, 1]
                        hc_pd_val_pred.extend(preds_hc.cpu().numpy())
                        hc_pd_val_labels.extend(valid_labels_hc.cpu().numpy())
                        hc_pd_val_probs.extend(probs_hc.cpu().numpy())
                    
                    # PD vs DD loss  
                    valid_pd_dd_mask = (pd_dd != -1)
                    if valid_pd_dd_mask.any():
                        valid_logits_pd = pd_dd_logits[valid_pd_dd_mask]
                        valid_labels_pd = pd_dd[valid_pd_dd_mask]
                        loss_pd = pd_dd_loss(valid_logits_pd, valid_labels_pd)
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
            
            val_loss /= len(val_loader)
            
            if epoch == config['num_epochs'] - 1 and epoch_features:
                fold_features = np.concatenate(epoch_features, axis=0)
                fold_hc_pd_labels = np.array(epoch_hc_pd_labels)
                fold_pd_dd_labels = np.array(epoch_pd_dd_labels)
            
            print("\n" + "="*60)
            val_metrics_hc = calculate_metrics(hc_pd_val_labels, hc_pd_val_pred, 
                                             f"{'Fold ' + str(fold_idx+1) + ' ' if num_folds > 1 else ''}Validation HC vs PD", 
                                             verbose=True)
            val_metrics_pd = calculate_metrics(pd_dd_val_labels, pd_dd_val_pred, 
                                             f"{'Fold ' + str(fold_idx+1) + ' ' if num_folds > 1 else ''}Validation PD vs DD", 
                                             verbose=True)
            print("="*60)
            
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
            
            print(f"\n{'Fold ' + str(fold_idx+1) + ', ' if num_folds > 1 else ''}Epoch {epoch+1} Summary:")
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Train Acc - HC vs PD: {train_acc_hc:.4f}, PD vs DD: {train_acc_pd:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            print(f"Val Acc - HC vs PD: {val_acc_hc:.4f}, PD vs DD: {val_acc_pd:.4f}, Combined: {val_acc_combined:.4f}")
            
            # Save best model and store probabilities for ROC
            if val_acc_combined > best_val_acc:
                best_val_acc = val_acc_combined
                best_epoch = epoch + 1
                
                # Store best predictions and probabilities for ROC curves
                if hc_pd_val_probs:
                    best_hc_pd_probs = np.array(hc_pd_val_probs)
                    best_hc_pd_preds = np.array(hc_pd_val_pred)
                    best_hc_pd_labels = np.array(hc_pd_val_labels)
                
                if pd_dd_val_probs:
                    best_pd_dd_probs = np.array(pd_dd_val_probs)
                    best_pd_dd_preds = np.array(pd_dd_val_pred)
                    best_pd_dd_labels = np.array(pd_dd_val_labels)
                
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
        
        if config.get('save_metrics', True):
            fold_suffix = f"_fold_{fold_idx+1}" if num_folds > 1 else ""
            
            if fold_metrics_hc and fold_metrics_pd:
                save_fold_metric(fold_idx, fold_suffix, best_epoch, best_val_acc,
                     fold_metrics_hc, fold_metrics_pd)
                
        fold_result = {
            'best_val_accuracy': best_val_acc,
            'history': history,
            'features': fold_features,
            'hc_pd_labels': fold_hc_pd_labels,
            'pd_dd_labels': fold_pd_dd_labels
        }
        all_fold_results.append(fold_result)
        
        if config.get('create_plots', True):
            plot_dir = f"plots/{'fold_' + str(fold_idx+1) if num_folds > 1 else 'single_run'}"
            os.makedirs(plot_dir, exist_ok=True)
            
            plot_loss(history, f"{plot_dir}/loss.png")
            
            if best_hc_pd_probs is not None and len(best_hc_pd_labels) > 0:
                plot_roc_curves(best_hc_pd_labels, best_hc_pd_preds, best_hc_pd_probs, 
                              f"{plot_dir}/roc_hc_vs_pd.png")
            
            if best_pd_dd_probs is not None and len(best_pd_dd_labels) > 0:
                plot_roc_curves(best_pd_dd_labels, best_pd_dd_preds, best_pd_dd_probs, 
                              f"{plot_dir}/roc_pd_vs_dd.png")
            
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

        'batch_size': 30,
        'learning_rate': 0.0005,  
        'weight_decay': 0.01,
        'num_epochs': 100,
        'num_workers': 0,

        'save_metrics': True,
        'create_plots': True,
    }
    results = train_model(config)
    
    if results is None:
        print("Training failed due to data loading issues or diagnostics failure.")
        return None
    
    return results


if __name__ == "__main__":
    results = main()