import os
import numpy as np
import json
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
def calculate_metrics(y_true, y_pred, task_name="", verbose=True):
    """Calculate and display detailed metrics"""
    if len(y_true) == 0:
        return {}
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
    precision_avg, recall_avg, f1_avg, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
    
    # Confusion matrix
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
        print(f"Weighted Precision: {precision_avg:.4f}")
        print(f"Weighted Recall: {recall_avg:.4f}")
        print(f"Weighted F1: {f1_avg:.4f}")
        
        # Per-class metrics
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