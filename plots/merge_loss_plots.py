from PIL import Image, ImageOps
import os

ref_img_path = r"d:\Documents\parkinsons\results-new\label_efficiency_curve.png"
img1_path = r"d:\Documents\parkinsons\results-new\BaseModel_woutBandPass_wDownsampling\plots\fold_3\loss.png"
img2_path = r"d:\Documents\parkinsons\results-new\BaseModel_wBandPass_wDownsampling\plots\fold_2\loss.png"
out_path = r"d:\Documents\parkinsons\results-new\merged_loss_plots.pdf"

ref_img = Image.open(ref_img_path)
target_w, target_h = ref_img.size

print(f"Target size: {target_w}x{target_h}")

img1 = Image.open(img1_path).convert("RGB")
img2 = Image.open(img2_path).convert("RGB")

half_w = target_w // 2

# We will pad the images to fit into half_w x target_h, preserving aspect ratio with a white background
img1_fitted = ImageOps.pad(img1, (half_w, target_h), color=(255, 255, 255))
img2_fitted = ImageOps.pad(img2, (target_w - half_w, target_h), color=(255, 255, 255))

combined = Image.new("RGB", (target_w, target_h), (255, 255, 255))
combined.paste(img1_fitted, (0, 0))
combined.paste(img2_fitted, (half_w, 0))

combined.save(out_path, "PDF", resolution=300)
print(f"Saved merged PDF to: {out_path}")
