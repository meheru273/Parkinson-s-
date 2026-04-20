from PIL import Image

img_paths = [
    "D:/Documents/parkinsons/results-new/BaseModel_wBandPass_wDownsampling/plots/fold_5/tsne_hc_vs_pd.png",
    "D:/Documents/parkinsons/results-new/BaseModel_wBandPass_wDownsampling/plots/fold_5/tsne_pd_vs_dd.png",
    "D:/Documents/parkinsons/results-new/ThreeClass_BaseModel/output_3class/plots/fold_5/tsne_three_class.png"
]

images = [Image.open(p).convert("RGB") for p in img_paths]

# Resize all to same height
target_height = min(img.height for img in images)
resized = []

for img in images:
    ratio = target_height / img.height
    new_width = int(img.width * ratio)
    resized.append(img.resize((new_width, target_height), Image.LANCZOS))

# Combine horizontally (no gaps)
total_width = sum(img.width for img in resized)

combined = Image.new("RGB", (total_width, target_height), (255, 255, 255))

x_offset = 0
for img in resized:
    combined.paste(img, (x_offset, 0))
    x_offset += img.width

# Save as high-quality PDF
combined.save("tsne_combined.pdf", "PDF", resolution=300)

print("✅ Compact PDF created!")