import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from PIL import Image
import numpy as np
import os
import json


# Load the SAM2 model
checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))

box_path = "./box_points"
image_path = './uploads'
save_path = "./masked_outputs"

os.makedirs(save_path, exist_ok=True)

image_files = os.listdir(image_path)
box_paths = os.listdir(box_path)

with open("./box_points/annotations.json", "r") as file:
    data = json.load(file)

for individual_image in image_files:
    image_data_path = "/static/" + individual_image

    if image_data_path in data["annotations"]:
        bounding_box = data["annotations"][image_data_path]["boundingBoxes"][0]
        print(bounding_box)

    image = Image.open("./uploads/" + individual_image).convert("RGB")

    left = bounding_box['minX'] 
    upper = bounding_box['minY']
    right = left + bounding_box['width']
    lower = upper + bounding_box['height']

    cropped_image = image.crop((left, upper, right, lower))
    cropped_image_np = np.array(cropped_image)
    cropped_image.show()

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        predictor.set_image(cropped_image_np)
        masks, _, _ = predictor.predict()

    mask_overlay = np.zeros_like(cropped_image_np)

    for mask in masks:
        mask_overlay[mask == 1] = [255, 0, 0]
    
    mask_overlay_image = Image.fromarray(mask_overlay)

    # Blend the original image with the mask overlay (no labels or bounding boxes)
    blended_image = Image.blend(Image.fromarray(cropped_image_np), mask_overlay_image, alpha=0.5)

    # Save the image with masks
    blended_image_name = f"masked_{individual_image}"
    blended_image.save(os.path.join(save_path, blended_image_name))

    # Optionally display the result
    blended_image.show()





