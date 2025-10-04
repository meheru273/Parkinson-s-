import os
import numpy as np
import json
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.manifold import TSNE
################EvaluationFunctions###########
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
        print(f" Precision: {precision_avg:.4f}")
        print(f" Recall: {recall_avg:.4f}")
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

def save_metrics(y_true, y_pred, epoch, out_path="metrics.txt", label_names=None, append=False):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    if y_true.size == 0:
        raise ValueError("y_true is empty")

    labels = np.unique(np.concatenate([y_true, y_pred]))
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, sup = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)


    lines = []
    lines.append(f"------ epoch : {epoch} -------")
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
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, mode) as f:
        f.write("\n".join(lines) + "\n")
    return out_path

def plot_loss(history,output_path):
    epochs = range(1, len(history['train_loss']) + 1)
    plt.figure(figsize=(8,6))
    plt.plot(epochs, history['train_loss'], label='Training Loss')
    plt.plot(epochs, history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training vs Validation Loss')
    plt.savefig(output_path)
    
    
def save_fold_metric(fold_idx, fold_suffix, best_epoch, best_val_acc,
                     fold_metrics_hc, fold_metrics_pd):
    # Save HC vs PD metrics
    if fold_metrics_hc:
        hc_filename = f"metrics/hc_vs_pd_metrics{fold_suffix}.txt"
        with open(hc_filename, 'w') as f:
            f.write(f"{'='*70}\n")
            f.write(f"{'FOLD ' + str(fold_idx+1) + ' ' if fold_idx is not None else ''}HC vs PD METRICS - ALL EPOCHS\n")
            f.write(f"Best Epoch: {best_epoch} (Combined Accuracy: {best_val_acc:.4f})\n")
            f.write(f"{'='*70}\n\n")
            
            for epoch_data in fold_metrics_hc:
                f.write(f"EPOCH {epoch_data['epoch']}:\n")
                f.write(f"Accuracy: {epoch_data['metrics'].get('accuracy', 0):.4f}\n")
                f.write(f"Precision: {epoch_data['metrics'].get('precision', 0):.4f}\n")
                f.write(f"Recall: {epoch_data['metrics'].get('recall', 0):.4f}\n")
                f.write(f"F1-Score: {epoch_data['metrics'].get('f1', 0):.4f}\n")
                
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
            f.write(f"{'FOLD ' + str(fold_idx+1) + ' ' if fold_idx is not None else ''}PD vs DD METRICS - ALL EPOCHS\n")
            f.write(f"Best Epoch: {best_epoch} (Combined Accuracy: {best_val_acc:.4f})\n")
            f.write(f"{'='*70}\n\n")
            
            for epoch_data in fold_metrics_pd:
                f.write(f"EPOCH {epoch_data['epoch']}:\n")
                f.write(f"Accuracy: {epoch_data['metrics'].get('accuracy', 0):.4f}\n")
                f.write(f"Precision: {epoch_data['metrics'].get('precision', 0):.4f}\n")
                f.write(f"Recall: {epoch_data['metrics'].get('recall', 0):.4f}\n")
                f.write(f"F1-Score: {epoch_data['metrics'].get('f1', 0):.4f}\n")
                
                if len(epoch_data['labels']) > 0:
                    cm = confusion_matrix(epoch_data['labels'], epoch_data['predictions'])
                    f.write(f"Confusion Matrix:\n{cm}\n")
                f.write("-" * 50 + "\n\n")
        
        print(f"✓ PD vs DD metrics saved: {pd_filename}")


def plot_roc_curves(labels, predictions, probabilities, output_path):
    plt.figure(figsize=(10, 8))
    
    # Calculate ROC curve and AUC
    fpr, tpr, _ = roc_curve(labels, probabilities)
    roc_auc = auc(fpr, tpr)
    
    # Plot
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
    
    # Perform t-SNE dimensionality reduction
    print("Performing t-SNE dimensionality reduction...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    features_2d = tsne.fit_transform(features)
    
    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
   
    ax1 = axes[0]
    valid_hc_pd = hc_pd_labels != -1
    if np.any(valid_hc_pd):
        features_hc_pd = features_2d[valid_hc_pd]
        labels_hc_pd = hc_pd_labels[valid_hc_pd]
        

        colors_hc_pd = ['blue' if l == 0 else 'red' for l in labels_hc_pd]
        
        hc_mask = labels_hc_pd == 0
        if np.any(hc_mask):
            ax1.scatter(features_hc_pd[hc_mask, 0], features_hc_pd[hc_mask, 1], 
                       c='blue', label='HC', alpha=0.6, s=30, edgecolors='black', linewidth=0.5)
        
        pd_mask = labels_hc_pd == 1
        if np.any(pd_mask):
            ax1.scatter(features_hc_pd[pd_mask, 0], features_hc_pd[pd_mask, 1], 
                       c='red', label='PD', alpha=0.6, s=30, edgecolors='black', linewidth=0.5)
        
        ax1.set_title('t-SNE: HC vs PD Classification', fontsize=14, fontweight='bold')
        ax1.set_xlabel('t-SNE Component 1', fontsize=12)
        ax1.set_ylabel('t-SNE Component 2', fontsize=12)
        ax1.legend(loc='upper right', fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        n_hc = np.sum(hc_mask)
        n_pd = np.sum(pd_mask)
        ax1.text(0.02, 0.98, f'HC: {n_hc}\nPD: {n_pd}', 
                transform=ax1.transAxes, fontsize=10,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    else:
        ax1.text(0.5, 0.5, 'No HC vs PD data available', 
                ha='center', va='center', transform=ax1.transAxes)
        ax1.set_title('t-SNE: HC vs PD Classification', fontsize=14, fontweight='bold')
    

    ax2 = axes[1]
    valid_pd_dd = pd_dd_labels != -1
    if np.any(valid_pd_dd):
        features_pd_dd = features_2d[valid_pd_dd]
        labels_pd_dd = pd_dd_labels[valid_pd_dd]
        
    
        colors_pd_dd = ['green' if l == 0 else 'orange' for l in labels_pd_dd]
        
        pd_mask = labels_pd_dd == 0
        if np.any(pd_mask):
            ax2.scatter(features_pd_dd[pd_mask, 0], features_pd_dd[pd_mask, 1], 
                       c='green', label='PD', alpha=0.6, s=30, edgecolors='black', linewidth=0.5)
        
    
        dd_mask = labels_pd_dd == 1
        if np.any(dd_mask):
            ax2.scatter(features_pd_dd[dd_mask, 0], features_pd_dd[dd_mask, 1], 
                       c='orange', label='DD', alpha=0.6, s=30, edgecolors='black', linewidth=0.5)
        
        ax2.set_title('t-SNE: PD vs DD Classification', fontsize=14, fontweight='bold')
        ax2.set_xlabel('t-SNE Component 1', fontsize=12)
        ax2.set_ylabel('t-SNE Component 2', fontsize=12)
        ax2.legend(loc='upper right', fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        n_pd = np.sum(pd_mask)
        n_dd = np.sum(dd_mask)
        ax2.text(0.02, 0.98, f'PD: {n_pd}\nDD: {n_dd}', 
                transform=ax2.transAxes, fontsize=10,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    else:
        ax2.text(0.5, 0.5, 'No PD vs DD data available', 
                ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('t-SNE: PD vs DD Classification', fontsize=14, fontweight='bold')
    

    fig.suptitle('t-SNE Visualization of Feature Space', fontsize=16, fontweight='bold', y=1.02)
    
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'tsne_visualization.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"t-SNE plot saved to {output_path}")
    
   
    if np.any(valid_hc_pd):
        plt.figure(figsize=(8, 6))
        features_hc_pd = features_2d[valid_hc_pd]
        labels_hc_pd = hc_pd_labels[valid_hc_pd]
        
        hc_mask = labels_hc_pd == 0
        pd_mask = labels_hc_pd == 1
        
        if np.any(hc_mask):
            plt.scatter(features_hc_pd[hc_mask, 0], features_hc_pd[hc_mask, 1], 
                       c='blue', label=f'HC (n={np.sum(hc_mask)})', 
                       alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
        
        if np.any(pd_mask):
            plt.scatter(features_hc_pd[pd_mask, 0], features_hc_pd[pd_mask, 1], 
                       c='red', label=f'PD (n={np.sum(pd_mask)})', 
                       alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
        
        plt.title('t-SNE: Healthy Control vs Parkinson\'s Disease', fontsize=14, fontweight='bold')
        plt.xlabel('t-SNE Component 1', fontsize=12)
        plt.ylabel('t-SNE Component 2', fontsize=12)
        plt.legend(loc='best', fontsize=11)
        plt.grid(True, alpha=0.3)
        
        output_path_hc = os.path.join(output_dir, 'tsne_hc_vs_pd.png')
        plt.savefig(output_path_hc, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"HC vs PD t-SNE plot saved to {output_path_hc}")
    

    if np.any(valid_pd_dd):
        plt.figure(figsize=(8, 6))
        features_pd_dd = features_2d[valid_pd_dd]
        labels_pd_dd = pd_dd_labels[valid_pd_dd]
        
        pd_mask = labels_pd_dd == 0
        dd_mask = labels_pd_dd == 1
        
        if np.any(pd_mask):
            plt.scatter(features_pd_dd[pd_mask, 0], features_pd_dd[pd_mask, 1], 
                       c='green', label=f'PD (n={np.sum(pd_mask)})', 
                       alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
        
        if np.any(dd_mask):
            plt.scatter(features_pd_dd[dd_mask, 0], features_pd_dd[dd_mask, 1], 
                       c='orange', label=f'DD (n={np.sum(dd_mask)})', 
                       alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
        
        plt.title('t-SNE: Parkinson\'s Disease vs Differential Diagnosis', fontsize=14, fontweight='bold')
        plt.xlabel('t-SNE Component 1', fontsize=12)
        plt.ylabel('t-SNE Component 2', fontsize=12)
        plt.legend(loc='best', fontsize=11)
        plt.grid(True, alpha=0.3)
        
        output_path_pd = os.path.join(output_dir, 'tsne_pd_vs_dd.png')
        plt.savefig(output_path_pd, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"PD vs DD t-SNE plot saved to {output_path_pd}")
    
    return features_2d
