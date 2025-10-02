import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os

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
