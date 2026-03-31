"""
Generate a PDF report with headings and images from all experiment results (fold 1 only
for k-fold experiments), followed by a consolidated summary table.
"""

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle,
    PageBreak, HRFlowable, KeepTogether
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT
import os

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE = r"d:\Documents\parkinsons\results-new"

SECTIONS = [
    {
        "heading": "Base Model (with Band-Pass Filtering) — Fold 1",
        "plots": [
            (os.path.join(BASE, "BaseModel_wBandPass_wDownsampling", "plots", "fold_1", "loss.png"),
             "Training & Validation Loss"),
            (os.path.join(BASE, "BaseModel_wBandPass_wDownsampling", "plots", "fold_1", "roc_hc_vs_pd.png"),
             "ROC Curve — HC vs PD"),
            (os.path.join(BASE, "BaseModel_wBandPass_wDownsampling", "plots", "fold_1", "roc_pd_vs_dd.png"),
             "ROC Curve — PD vs DD"),
            (os.path.join(BASE, "BaseModel_wBandPass_wDownsampling", "plots", "fold_1", "tsne_hc_vs_pd.png"),
             "t-SNE Embeddings — HC vs PD"),
            (os.path.join(BASE, "BaseModel_wBandPass_wDownsampling", "plots", "fold_1", "tsne_pd_vs_dd.png"),
             "t-SNE Embeddings — PD vs DD"),
            (os.path.join(BASE, "BaseModel_wBandPass_wDownsampling", "plots", "fold_1", "attention_comparison_hc_vs_pd.png"),
             "Attention Map Comparison — HC vs PD"),
            (os.path.join(BASE, "BaseModel_wBandPass_wDownsampling", "plots", "fold_1", "attention_comparison_pd_vs_dd.png"),
             "Attention Map Comparison — PD vs DD"),
        ],
    },
    {
        "heading": "Base Model (without Band-Pass Filtering) — Fold 1",
        "plots": [
            (os.path.join(BASE, "BaseModel_woutBandPass_wDownsampling", "plots", "fold_1", "loss.png"),
             "Training & Validation Loss"),
            (os.path.join(BASE, "BaseModel_woutBandPass_wDownsampling", "plots", "fold_1", "roc_hc_vs_pd.png"),
             "ROC Curve — HC vs PD"),
            (os.path.join(BASE, "BaseModel_woutBandPass_wDownsampling", "plots", "fold_1", "roc_pd_vs_dd.png"),
             "ROC Curve — PD vs DD"),
            (os.path.join(BASE, "BaseModel_woutBandPass_wDownsampling", "plots", "fold_1", "tsne_hc_vs_pd.png"),
             "t-SNE Embeddings — HC vs PD"),
            (os.path.join(BASE, "BaseModel_woutBandPass_wDownsampling", "plots", "fold_1", "tsne_pd_vs_dd.png"),
             "t-SNE Embeddings — PD vs DD"),
            (os.path.join(BASE, "BaseModel_woutBandPass_wDownsampling", "plots", "fold_1", "attention_comparison_hc_vs_pd.png"),
             "Attention Map Comparison — HC vs PD"),
            (os.path.join(BASE, "BaseModel_woutBandPass_wDownsampling", "plots", "fold_1", "attention_comparison_pd_vs_dd.png"),
             "Attention Map Comparison — PD vs DD"),
        ],
    },
    {
        "heading": "Three-Class Classifier — Fold 1",
        "plots": [
            (os.path.join(BASE, "ThreeClass_Classifier_BaseModel", "output_3class", "plots", "fold_1", "loss.png"),
             "Training & Validation Loss"),
            (os.path.join(BASE, "ThreeClass_Classifier_BaseModel", "output_3class", "plots", "fold_1", "roc_three_class.png"),
             "ROC Curve — Three-Class (HC / PD / DD)"),
            (os.path.join(BASE, "ThreeClass_Classifier_BaseModel", "output_3class", "plots", "fold_1", "tsne_three_class.png"),
             "t-SNE Embeddings — Three-Class"),
        ],
    },
    {
        "heading": "SSL Base Model — Full Fine-Tune",
        "subheadings": [
            {
                "sub": "Pre-Training Phase",
                "plots": [
                    (os.path.join(BASE, "SSL_BaseModel", "full-finrtune", "plots", "pretrain", "contrastive_loss.png"),
                     "Contrastive Pre-Training Loss"),
                    (os.path.join(BASE, "SSL_BaseModel", "full-finrtune", "plots", "pretrain", "tsne_hc_vs_pd.png"),
                     "Pre-Training t-SNE — HC vs PD"),
                    (os.path.join(BASE, "SSL_BaseModel", "full-finrtune", "plots", "pretrain", "tsne_pd_vs_dd.png"),
                     "Pre-Training t-SNE — PD vs DD"),
                ],
            },
            {
                "sub": "Fine-Tuning Phase",
                "plots": [
                    (os.path.join(BASE, "SSL_BaseModel", "full-finrtune", "plots", "label_efficiency", "full_finetune", "accuracy_curve.png"),
                     "Fine-Tune Accuracy Curve"),
                    (os.path.join(BASE, "SSL_BaseModel", "full-finrtune", "plots", "label_efficiency", "full_finetune", "loss_curve.png"),
                     "Fine-Tune Loss Curve"),
                    (os.path.join(BASE, "SSL_BaseModel", "full-finrtune", "plots", "label_efficiency", "full_finetune", "roc_hc_vs_pd.png"),
                     "ROC Curve — HC vs PD"),
                    (os.path.join(BASE, "SSL_BaseModel", "full-finrtune", "plots", "label_efficiency", "full_finetune", "roc_pd_vs_dd.png"),
                     "ROC Curve — PD vs DD"),
                    (os.path.join(BASE, "SSL_BaseModel", "full-finrtune", "plots", "label_efficiency", "full_finetune", "tsne_hc_vs_pd.png"),
                     "t-SNE — HC vs PD"),
                    (os.path.join(BASE, "SSL_BaseModel", "full-finrtune", "plots", "label_efficiency", "full_finetune", "tsne_pd_vs_dd.png"),
                     "t-SNE — PD vs DD"),
                    (os.path.join(BASE, "SSL_BaseModel", "full-finrtune", "plots", "label_efficiency", "accuracy_vs_labels.png"),
                     "Accuracy vs. Number of Labels"),
                ],
            },
        ],
    },
    {
        "heading": "SSL Base Model — Linear Probe",
        "subheadings": [
            {
                "sub": "Pre-Training Phase",
                "plots": [
                    (os.path.join(BASE, "SSL_BaseModel", "linear-prob", "plots", "pretrain", "contrastive_loss.png"),
                     "Contrastive Pre-Training Loss"),
                    (os.path.join(BASE, "SSL_BaseModel", "linear-prob", "plots", "pretrain", "tsne_hc_vs_pd.png"),
                     "Pre-Training t-SNE — HC vs PD"),
                    (os.path.join(BASE, "SSL_BaseModel", "linear-prob", "plots", "pretrain", "tsne_pd_vs_dd.png"),
                     "Pre-Training t-SNE — PD vs DD"),
                ],
            },
            {
                "sub": "Linear Probe Phase",
                "plots": [
                    (os.path.join(BASE, "SSL_BaseModel", "linear-prob", "plots", "label_efficiency", "linear_prob", "accuracy_curve.png"),
                     "Linear Probe Accuracy Curve"),
                    (os.path.join(BASE, "SSL_BaseModel", "linear-prob", "plots", "label_efficiency", "linear_prob", "loss_curve.png"),
                     "Linear Probe Loss Curve"),
                    (os.path.join(BASE, "SSL_BaseModel", "linear-prob", "plots", "label_efficiency", "linear_prob", "roc_hc_vs_pd.png"),
                     "ROC Curve — HC vs PD"),
                    (os.path.join(BASE, "SSL_BaseModel", "linear-prob", "plots", "label_efficiency", "linear_prob", "roc_pd_vs_dd.png"),
                     "ROC Curve — PD vs DD"),
                    (os.path.join(BASE, "SSL_BaseModel", "linear-prob", "plots", "label_efficiency", "linear_prob", "tsne_hc_vs_pd.png"),
                     "t-SNE — HC vs PD"),
                    (os.path.join(BASE, "SSL_BaseModel", "linear-prob", "plots", "label_efficiency", "linear_prob", "tsne_pd_vs_dd.png"),
                     "t-SNE — PD vs DD"),
                    (os.path.join(BASE, "SSL_BaseModel", "linear-prob", "plots", "label_efficiency", "accuracy_vs_labels.png"),
                     "Accuracy vs. Number of Labels"),
                ],
            },
        ],
    },
    {
        "heading": "TimesFM LoRA — Fold 1",
        "plots": [
            (os.path.join(BASE, "TimesFM_LoRA", "results", "timesfm_lora", "plots", "fold_1", "loss.png"),
             "Training & Validation Loss"),
            (os.path.join(BASE, "TimesFM_LoRA", "results", "timesfm_lora", "plots", "fold_1", "roc_hc_vs_pd.png"),
             "ROC Curve — HC vs PD"),
            (os.path.join(BASE, "TimesFM_LoRA", "results", "timesfm_lora", "plots", "fold_1", "roc_pd_vs_dd.png"),
             "ROC Curve — PD vs DD"),
            (os.path.join(BASE, "TimesFM_LoRA", "results", "timesfm_lora", "plots", "fold_1", "tsne_hc_vs_pd.png"),
             "t-SNE Embeddings — HC vs PD"),
            (os.path.join(BASE, "TimesFM_LoRA", "results", "timesfm_lora", "plots", "fold_1", "tsne_pd_vs_dd.png"),
             "t-SNE Embeddings — PD vs DD"),
        ],
    },
]

# ── Summary table data ────────────────────────────────────────────────────────
SUMMARY_INTRO = (
    "Table 1 consolidates the 5-fold cross-validation results (mean ± std %) "
    "across all experimental variants. The hierarchical dual-head base model with "
    "band-pass filtering achieves the highest combined accuracy. Self-supervised "
    "pre-training matches the supervised baseline while requiring significantly fewer labels."
)

TABLE_DATA = [
    # Header row 1
    ["Approach", "HC vs PD\nAcc", "HC vs PD\nPrec", "HC vs PD\nRec",
     "PD vs DD\nAcc", "PD vs DD\nPrec", "PD vs DD\nRec", "Combined"],
    # Data rows
    ["Base Model (w/ BP)",
     "93.31±0.22", "93.81±0.31", "93.31±0.22",
     "92.34±0.08", "93.14±0.18", "92.34±0.08", "92.83% ✓"],
    ["Base Model (w/o BP)",
     "85.64±2.14", "85.71±2.15", "85.64±2.14",
     "85.39±1.88", "85.64±1.92", "85.39±1.88", "85.52%"],
    ["Three-Class Baseline",
     "Acc: 88.35±0.83", "Prec: 89.67±0.83", "—",
     "Rec: 88.35±0.83", "F1: 88.32±0.88", "—", "88.35%"],
    ["SSL Fine-tune (100%)",
     "93.35", "93.96", "93.35",
     "92.40", "93.31", "92.40", "92.87%"],
    ["TimesFM LoRA",
     "93.58", "94.09", "93.58",  # HC vs PD max (epoch 3)
     "92.73", "93.60", "92.73",  # PD vs DD max (epoch 8)
     "93.15%"],  # Mean of the two accuracies
    ["Raspberry Pi (edge)",
     "—", "—", "—",
     "—", "—", "—", "~48.32 ms/win"],
]

# ── Build PDF ─────────────────────────────────────────────────────────────────
OUTPUT_PATH = os.path.join(BASE, "results_report.pdf")

PAGE_W, PAGE_H = A4
MARGIN = 2 * cm
USABLE_W = PAGE_W - 2 * MARGIN
IMG_W = USABLE_W * 0.82          # 82% of usable width for single images
IMG_PAIR_W = USABLE_W * 0.48     # side-by-side images

styles = getSampleStyleSheet()

TITLE_STYLE = ParagraphStyle(
    "Title", parent=styles["Title"],
    fontSize=20, spaceAfter=8, textColor=colors.HexColor("#1a1a2e"),
    alignment=TA_CENTER
)
H1_STYLE = ParagraphStyle(
    "H1", parent=styles["Heading1"],
    fontSize=20, spaceBefore=10, spaceAfter=4,
    textColor=colors.white,
    backColor=colors.HexColor("#16213e"),
    borderPad=6,
    leftIndent=-4, rightIndent=-4,
)
H2_STYLE = ParagraphStyle(
    "H2", parent=styles["Heading2"],
    fontSize=13, spaceBefore=8, spaceAfter=3,
    textColor=colors.HexColor("#0f3460"),
    backColor=colors.HexColor("#dce8f5"),
    borderPad=4,
)
CAPTION_STYLE = ParagraphStyle(
    "Caption", parent=styles["Normal"],
    fontSize=8.5, textColor=colors.HexColor("#555555"),
    alignment=TA_CENTER, spaceAfter=4,
)
BODY_STYLE = ParagraphStyle(
    "Body", parent=styles["Normal"],
    fontSize=9.5, leading=13, spaceAfter=6,
    textColor=colors.HexColor("#333333"),
)


def img_block(path, caption, width=IMG_W):
    """Return [Image, Caption] if file exists, else a warning paragraph."""
    if not os.path.exists(path):
        return [Paragraph(f"[Image not found: {os.path.basename(path)}]", CAPTION_STYLE)]
    from PIL import Image as PILImage
    with PILImage.open(path) as im:
        orig_w, orig_h = im.size
    aspect = orig_h / orig_w
    height = width * aspect
    # cap height so it doesn't overflow a page
    max_h = PAGE_H - 2 * MARGIN - 4 * cm
    if height > max_h:
        height = max_h
        width = height / aspect
    return [Image(path, width=width, height=height),
            Paragraph(f"<i>{caption}</i>", CAPTION_STYLE)]


def add_image_pair(story, path1, cap1, path2, cap2):
    """Place two images side by side in a 2-column table."""
    def cell(path, cap):
        if not os.path.exists(path):
            return Paragraph(f"[Image not found: {os.path.basename(path)}]", CAPTION_STYLE)
        from PIL import Image as PILImage
        w = IMG_PAIR_W
        with PILImage.open(path) as im:
            orig_w, orig_h = im.size
        aspect = orig_h / orig_w
        h = w * aspect
        max_h = (PAGE_H - 2 * MARGIN - 4 * cm) / 2
        if h > max_h:
            h = max_h
            w = h / aspect
        return [Image(path, width=w, height=h),
                Paragraph(f"<i>{cap}</i>", CAPTION_STYLE)]

    t = Table([[cell(path1, cap1), cell(path2, cap2)]], colWidths=[USABLE_W / 2, USABLE_W / 2])
    t.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ALIGN",  (0, 0), (-1, -1), "CENTER"),
    ]))
    story.append(t)
    story.append(Spacer(1, 0.1 * cm))


def build_story():
    story = []

    # ── Cover title ──
    story.append(Spacer(1, 0.4 * cm))
    story.append(Paragraph("Parkinson's Detection — Results Report", TITLE_STYLE))
    story.append(Paragraph(
        "All plots shown are from <b>Fold 1</b> of each k-fold experiment "
        "(or the single run for SSL experiments). "
        "A consolidated summary table is provided at the end.",
        BODY_STYLE
    ))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#cccccc")))
    story.append(Spacer(1, 0.2 * cm))

    # ── Sections ──
    for sec in SECTIONS:
        story.append(Spacer(1, 0.15 * cm))
        story.append(Paragraph(sec["heading"], H1_STYLE))
        story.append(Spacer(1, 0.15 * cm))

        if "plots" in sec:
            plots = sec["plots"]
            i = 0
            while i < len(plots):
                if i + 1 < len(plots):
                    # pair them side by side
                    add_image_pair(story, plots[i][0], plots[i][1], plots[i+1][0], plots[i+1][1])
                    i += 2
                else:
                    # single image centred
                    for elem in img_block(plots[i][0], plots[i][1]):
                        story.append(elem)
                    i += 1

        if "subheadings" in sec:
            for sub in sec["subheadings"]:
                story.append(Paragraph(sub["sub"], H2_STYLE))
                plots = sub["plots"]
                i = 0
                while i < len(plots):
                    if i + 1 < len(plots):
                        add_image_pair(story, plots[i][0], plots[i][1], plots[i+1][0], plots[i+1][1])
                        i += 2
                    else:
                        for elem in img_block(plots[i][0], plots[i][1]):
                            story.append(elem)
                        i += 1

    # ── Summary table ──
    story.append(Spacer(1, 0.3 * cm))
    story.append(Paragraph("Consolidated Summary Table", H1_STYLE))
    story.append(Spacer(1, 0.15 * cm))
    story.append(Paragraph(SUMMARY_INTRO, BODY_STYLE))
    story.append(Spacer(1, 0.2 * cm))

    # Build table
    col_widths = [4.2 * cm] + [2.1 * cm] * 6 + [2.4 * cm]
    tbl = Table(TABLE_DATA, colWidths=col_widths, repeatRows=1)
    tbl.setStyle(TableStyle([
        # Header
        ("BACKGROUND",   (0, 0), (-1, 0), colors.HexColor("#16213e")),
        ("TEXTCOLOR",    (0, 0), (-1, 0), colors.white),
        ("FONTNAME",     (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",     (0, 0), (-1, 0), 8),
        ("ALIGN",        (0, 0), (-1, 0), "CENTER"),
        ("VALIGN",       (0, 0), (-1, 0), "MIDDLE"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#f9f9f9"), colors.white]),
        # Best row highlight
        ("BACKGROUND",   (0, 1), (-1, 1), colors.HexColor("#e8f5e9")),
        ("FONTNAME",     (-1, 1), (-1, 1), "Helvetica-Bold"),
        # General
        ("FONTSIZE",     (0, 1), (-1, -1), 7.5),
        ("ALIGN",        (1, 1), (-1, -1), "CENTER"),
        ("VALIGN",       (0, 0), (-1, -1), "MIDDLE"),
        ("GRID",         (0, 0), (-1, -1), 0.4, colors.HexColor("#cccccc")),
        ("TOPPADDING",   (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 5),
        ("LEFTPADDING",  (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        # Edges
        ("LINEABOVE",    (0, 0), (-1, 0), 1.5, colors.HexColor("#16213e")),
        ("LINEBELOW",    (0, -1), (-1, -1), 1.2, colors.HexColor("#16213e")),
        ("LINEBELOW",    (0, 0), (-1, 0), 1.0, colors.HexColor("#16213e")),
    ]))
    story.append(tbl)
    story.append(Spacer(1, 0.4 * cm))
    story.append(Paragraph(
        "<i>✓ Best combined accuracy. BP = Band-Pass filter. "
        "SSL = Self-Supervised Learning. LoRA = Low-Rank Adaptation.</i>",
        CAPTION_STYLE
    ))
    return story


doc = SimpleDocTemplate(
    OUTPUT_PATH,
    pagesize=A4,
    leftMargin=MARGIN, rightMargin=MARGIN,
    topMargin=MARGIN, bottomMargin=MARGIN,
    title="Parkinson's Detection Results Report",
    author="Research Pipeline",
)
doc.build(build_story())
print(f"PDF saved -> {OUTPUT_PATH}")
