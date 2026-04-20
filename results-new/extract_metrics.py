import csv, os

base = r'd:\Documents\parkinsons\results-new\ThreeClass_BaseModel\metrics_3class'
for fold in range(1, 6):
    path = os.path.join(base, f'three_class_metrics_fold_{fold}.csv')
    with open(path) as f:
        rows = list(csv.DictReader(f))
    best = max(rows, key=lambda r: float(r['accuracy']))
    print(
        f"Fold {fold}: epoch={best['epoch']}, "
        f"acc={float(best['accuracy']):.4f}, "
        f"f1_HC={float(best['f1_HC']):.4f}, "
        f"f1_PD={float(best['f1_PD']):.4f}, "
        f"f1_DD={float(best['f1_DD']):.4f}, "
        f"acc_HC={float(best['accuracy_HC']):.4f}, "
        f"acc_PD={float(best['accuracy_PD']):.4f}, "
        f"acc_DD={float(best['accuracy_DD']):.4f}"
    )
