import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from PIL import Image
import numpy as np

# Load the SAM2 model
checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))

# Load the image
image_path = 'glass1.png'
image = Image.open(image_path).convert("RGB")
image_np = np.array(image)

# Run SAM2 to get all masks
with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    predictor.set_image(image_np)
    masks, _, _ = predictor.predict()

# Create a blank mask image for visualization
mask_overlay = np.zeros_like(image_np)

# Iterate over the masks and apply them to the mask overlay
for mask in masks:
    # Each mask is an array of 1s (mask) and 0s (background)
    mask_overlay[mask == 1] = [255, 0, 0]  # Red mask for visualization

# Convert the mask overlay to a PIL image
mask_overlay_image = Image.fromarray(mask_overlay)

# Blend the original image with the mask overlay (no labels or bounding boxes)
blended_image = Image.blend(Image.fromarray(image_np), mask_overlay_image, alpha=0.5)

# Save the image with masks
blended_image.save('glass1_with_masks.png')

# Optionally display the result
blended_image.show()
