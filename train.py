import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
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
from metrics import calculate_metrics, save_metrics

class EarlyStopping:
    """Enhanced early stopping with multiple metrics"""
    
    def __init__(self, patience=7, min_delta=0.001, monitor='f1'):
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.counter = 0
        self.best_score = -float('inf') if monitor == 'f1' else float('inf')
        self.is_better = (lambda new, best: new > best + min_delta) if monitor == 'f1' else (lambda new, best: new < best - min_delta)
        
    def __call__(self, current_score):
        if self.is_better(current_score, self.best_score):
            self.best_score = current_score
            self.counter = 0
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience

class BalancedFocalLoss(nn.Module):
    """Improved Focal Loss with better balance for medical data"""
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(BalancedFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        
        # Calculate alpha_t for each sample
        if self.alpha is not None:
            if isinstance(self.alpha, (float, int)):
                alpha_t = self.alpha
            else:
                alpha_t = self.alpha[targets]
        else:
            alpha_t = 1.0
        
        focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class LabelSmoothingCrossEntropy(nn.Module):
    """Label smoothing to prevent overconfident predictions"""
    def __init__(self, smoothing=0.1, weight=None):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing
        self.weight = weight
        
    def forward(self, pred, target):
        n_class = pred.size(1)
        one_hot = torch.zeros_like(pred).scatter(1, target.unsqueeze(1), 1)
        one_hot = one_hot * (1 - self.smoothing) + (1 - one_hot) * self.smoothing / (n_class - 1)
        log_prob = F.log_softmax(pred, dim=1)
        
        if self.weight is not None:
            weight_expanded = self.weight[target]
            loss = -(one_hot * log_prob).sum(dim=1) * weight_expanded
        else:
            loss = -(one_hot * log_prob).sum(dim=1)
        
        return loss.mean()

class ContrastiveLoss(nn.Module):
    """Contrastive learning for better feature separation"""
    def __init__(self, temperature=0.1):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
        
    def forward(self, features, labels):
        # Normalize features
        features = F.normalize(features, dim=1)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(features, features.T) / self.temperature
        
        # Create positive and negative masks
        labels = labels.unsqueeze(1)
        mask = torch.eq(labels, labels.T).float()
        
        # Remove diagonal elements
        mask = mask - torch.eye(mask.size(0), device=mask.device)
        
        # Compute contrastive loss
        exp_sim = torch.exp(similarity_matrix)
        sum_exp_sim = exp_sim.sum(dim=1, keepdim=True)
        
        positive_sim = (exp_sim * mask).sum(dim=1)
        
        loss = -torch.log(positive_sim / sum_exp_sim + 1e-8)
        return loss.mean()

def create_balanced_sampler(dataset):
    """Create a balanced sampler for training"""
    # Get labels for HC vs PD task (primary task)
    hc_vs_pd_labels = dataset.hc_vs_pd_left
    valid_indices = np.where(hc_vs_pd_labels != -1)[0]
    valid_labels = hc_vs_pd_labels[valid_indices]
    
    # Calculate weights for each class
    class_counts = np.bincount(valid_labels)
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[valid_labels]
    
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(valid_indices),
        replacement=True
    ), valid_indices

def train_model_advanced(config: Dict):
    """Advanced training with multiple techniques for medical AI"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load dataset
    full_dataset = ParkinsonsDataLoader(config['data_root'])
    train_dataset, val_dataset = full_dataset.get_train_test_split()
    
    # Print dataset statistics
    hc_count = np.sum(full_dataset.hc_vs_pd_left == 0)
    pd_count = np.sum(full_dataset.hc_vs_pd_left == 1)
    pd_dd_pd_count = np.sum(full_dataset.pd_vs_dd_left == 0)
    pd_dd_dd_count = np.sum(full_dataset.pd_vs_dd_left == 1)
    
    print(f"Dataset Statistics:")
    print(f"HC vs PD - HC: {hc_count}, PD: {pd_count} (ratio: {pd_count/hc_count:.2f})")
    print(f"PD vs DD - PD: {pd_dd_pd_count}, DD: {pd_dd_dd_count} (ratio: {pd_dd_pd_count/pd_dd_dd_count:.2f})")
    
    # Create balanced sampler for training
    sampler, valid_indices = create_balanced_sampler(train_dataset)
    
    # Data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        sampler=sampler,
        num_workers=config['num_workers']
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False,
        num_workers=config['num_workers']
    )
    
    # Model
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
    
    # Advanced optimizer with different learning rates for different parts
    backbone_params = []
    classifier_params = []
    
    for name, param in model.named_parameters():
        if 'head_' in name:
            classifier_params.append(param)
        else:
            backbone_params.append(param)
    
    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': config['learning_rate'] * 0.1},  # Lower LR for backbone
        {'params': classifier_params, 'lr': config['learning_rate']}        # Higher LR for classifiers
    ], weight_decay=config['weight_decay'])
    
    # Advanced scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-7
    )
    
    # Calculate balanced class weights
    hc_pd_ratio = pd_count / hc_count
    pd_dd_ratio = pd_dd_pd_count / pd_dd_dd_count
    
    hc_pd_weights = torch.FloatTensor([hc_pd_ratio, 1.0]).to(device)
    pd_dd_weights = torch.FloatTensor([1.0, pd_dd_ratio]).to(device)
    
    print(f"Class weights - HC vs PD: {hc_pd_weights}")
    print(f"Class weights - PD vs DD: {pd_dd_weights}")
    
    # Multiple loss functions
    criterion_hc_vs_pd = LabelSmoothingCrossEntropy(smoothing=0.1, weight=hc_pd_weights)
    criterion_pd_vs_dd = LabelSmoothingCrossEntropy(smoothing=0.1, weight=pd_dd_weights)
    
    # Contrastive loss for better feature separation
    contrastive_loss = ContrastiveLoss(temperature=0.1)
    
    # Early stopping based on F1 score
    early_stopping = EarlyStopping(patience=config['patience'], monitor='f1')
    
    # Training history
    history = defaultdict(list)
    best_f1_combined = 0.0
    
    print("Starting advanced training...")
    
    # Training loop
    for epoch in range(config['num_epochs']):
        print(f"\nEpoch {epoch+1}/{config['num_epochs']}")
        print("-" * 50)
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_preds_hc_vs_pd = []
        train_labels_hc_vs_pd = []
        train_preds_pd_vs_dd = []
        train_labels_pd_vs_dd = []
        
        for batch_idx, batch in enumerate(tqdm(train_loader, desc="Training")):
            left_sample, right_sample, hc_vs_pd_left, pd_vs_dd_left, hc_vs_pd_right, pd_vs_dd_right = batch
            
            left_sample = left_sample.to(device)
            right_sample = right_sample.to(device)
            hc_vs_pd_left = hc_vs_pd_left.to(device)
            pd_vs_dd_left = pd_vs_dd_left.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            logits_hc_vs_pd, logits_pd_vs_dd = model(left_sample, right_sample)
            
            # Get features for contrastive learning (from the fused features before classification)
            with torch.no_grad():
                # You'd need to modify your model to return features as well
                # For now, we'll use the logits as a proxy
                features = torch.cat([logits_hc_vs_pd, logits_pd_vs_dd], dim=1)
            
            total_loss = 0
            loss_count = 0
            
            # HC vs PD loss
            valid_hc_vs_pd_mask = (hc_vs_pd_left != -1)
            if valid_hc_vs_pd_mask.any():
                valid_logits_hc = logits_hc_vs_pd[valid_hc_vs_pd_mask]
                valid_labels_hc = hc_vs_pd_left[valid_hc_vs_pd_mask]
                
                # Classification loss
                loss_hc = criterion_hc_vs_pd(valid_logits_hc, valid_labels_hc)
                total_loss += loss_hc
                loss_count += 1
                
                # Contrastive loss (if we have enough samples of both classes)
                if len(torch.unique(valid_labels_hc)) > 1 and len(valid_labels_hc) > 4:
                    valid_features_hc = features[valid_hc_vs_pd_mask]
                    contrastive_hc = contrastive_loss(valid_features_hc, valid_labels_hc)
                    total_loss += 0.1 * contrastive_hc  # Small weight for contrastive loss
                
                preds_hc = torch.argmax(valid_logits_hc, dim=1)
                train_preds_hc_vs_pd.extend(preds_hc.cpu().numpy())
                train_labels_hc_vs_pd.extend(valid_labels_hc.cpu().numpy())
            
            # PD vs DD loss
            valid_pd_vs_dd_mask = (pd_vs_dd_left != -1)
            if valid_pd_vs_dd_mask.any():
                valid_logits_pd = logits_pd_vs_dd[valid_pd_vs_dd_mask]
                valid_labels_pd = pd_vs_dd_left[valid_pd_vs_dd_mask]
                
                # Classification loss  
                loss_pd = criterion_pd_vs_dd(valid_logits_pd, valid_labels_pd)
                total_loss += 2.0 * loss_pd  # Higher weight for harder PD vs DD task
                loss_count += 1
                
                # Contrastive loss
                if len(torch.unique(valid_labels_pd)) > 1 and len(valid_labels_pd) > 4:
                    valid_features_pd = features[valid_pd_vs_dd_mask]
                    contrastive_pd = contrastive_loss(valid_features_pd, valid_labels_pd)
                    total_loss += 0.2 * contrastive_pd  # Slightly higher weight for harder task
                
                preds_pd = torch.argmax(valid_logits_pd, dim=1)
                train_preds_pd_vs_dd.extend(preds_pd.cpu().numpy())
                train_labels_pd_vs_dd.extend(valid_labels_pd.cpu().numpy())
            
            if loss_count > 0:
                avg_loss = total_loss / loss_count
                avg_loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_loss += avg_loss.item()
            
            # Update learning rate
            scheduler.step(epoch + batch_idx / len(train_loader))
        
        train_loss /= len(train_loader)
        
        # Calculate training metrics
        train_metrics_hc = calculate_metrics(train_labels_hc_vs_pd, train_preds_hc_vs_pd, "Training HC vs PD", verbose=False)
        train_metrics_pd = calculate_metrics(train_labels_pd_vs_dd, train_preds_pd_vs_dd, "Training PD vs DD", verbose=False)
        
        # Validation phase (similar to before but with metrics focus)
        model.eval()
        val_loss = 0.0
        val_preds_hc_vs_pd = []
        val_labels_hc_vs_pd = []
        val_preds_pd_vs_dd = []
        val_labels_pd_vs_dd = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                left_sample, right_sample, hc_vs_pd_left, pd_vs_dd_left, hc_vs_pd_right, pd_vs_dd_right = batch
                
                left_sample = left_sample.to(device)
                right_sample = right_sample.to(device)
                hc_vs_pd_left = hc_vs_pd_left.to(device)
                pd_vs_dd_left = pd_vs_dd_left.to(device)
                
                logits_hc_vs_pd, logits_pd_vs_dd = model(left_sample, right_sample)
                
                # Calculate losses
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
                    total_loss += 2.0 * loss_pd
                    loss_count += 1
                    
                    preds_pd = torch.argmax(valid_logits_pd, dim=1)
                    val_preds_pd_vs_dd.extend(preds_pd.cpu().numpy())
                    val_labels_pd_vs_dd.extend(valid_labels_pd.cpu().numpy())
                
                if loss_count > 0:
                    avg_loss = total_loss / loss_count
                    val_loss += avg_loss.item()
        
        val_loss /= len(val_loader)
        
        # Calculate validation metrics with detailed display
        print("\n" + "="*60)
        val_metrics_hc = calculate_metrics(val_labels_hc_vs_pd, val_preds_hc_vs_pd, "Validation HC vs PD", verbose=True)
        val_metrics_pd = calculate_metrics(val_labels_pd_vs_dd, val_preds_pd_vs_dd, "Validation PD vs DD", verbose=True)
        print("="*60)
        
        # Focus on F1 scores for medical AI
        val_f1_hc = val_metrics_hc.get('f1_avg', 0)
        val_f1_pd = val_metrics_pd.get('f1_avg', 0)
        val_f1_combined = (val_f1_hc + val_f1_pd) / 2
        
        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_f1_hc'].append(val_f1_hc)
        history['val_f1_pd'].append(val_f1_pd)
        history['val_f1_combined'].append(val_f1_combined)
        
        # Print comprehensive metrics
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        print(f"Val F1 - HC vs PD: {val_f1_hc:.4f}, PD vs DD: {val_f1_pd:.4f}, Combined: {val_f1_combined:.4f}")
        print(f"Current LR: {scheduler.get_last_lr()[0]:.2e}")
        
        # Save best model based on F1 score
        if val_f1_combined > best_f1_combined:
            best_f1_combined = val_f1_combined
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'val_f1_combined': val_f1_combined,
                'val_f1_hc': val_f1_hc,
                'val_f1_pd': val_f1_pd,
                'config': config
            }, 'best_model_f1.pth')
            print("✓ New best F1 model saved!")
        
        # Early stopping based on F1 score
        if early_stopping(val_f1_combined):
            print(f"Early stopping at epoch {epoch+1} (F1 not improving)")
            break
    
    print(f"\nTraining completed! Best combined F1 score: {best_f1_combined:.4f}")
    
    return {
        'best_f1_combined': best_f1_combined,
        'history': history,
        'model': model
    }

def main():
    """Main function with advanced configuration"""
    
    config = {
        'data_root': "/kaggle/input/parkinsons/pads-parkinsons-disease-smartwatch-dataset-1.0.0",
        
        # Model architecture - optimized for medical data
        'input_dim': 6, 
        'd_model': 128,
        'num_heads': 8,
        'num_layers': 4,      # Increased depth for better pattern recognition
        'd_ff': 512,          # Increased capacity
        'dropout': 0.3,       # Higher dropout to prevent overfitting
        'seq_len': 256,
        'num_classes': 2,  
        
        # Training parameters - optimized for medical AI
        'batch_size': 12,     # Smaller batch for better gradient updates
        'learning_rate': 1e-4, # Lower learning rate for stability
        'weight_decay': 0.02,  # Higher regularization
        'num_epochs': 100,     # More epochs with early stopping
        'patience': 20,        # More patience for medical data
        'num_workers': 0,
    }
    
    print("Advanced Medical AI Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    results = train_model_advanced(config)
    
    if results is None:
        print("Training failed due to data loading issues.")
        return None
    
    return results

if __name__ == "__main__":
    results = main()