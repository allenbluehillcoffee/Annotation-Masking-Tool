import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from PIL import Image
import numpy as np

def run_sam2(bounding_box):
    checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
    model_cfg = "sam2/configs/sam2.1/sam2.1_hiera_l.yaml"
    predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))

    # Load the image
    image_path = ''
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)

    # Run SAM2 to get all masks
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        predictor.set_image(image_np)
        masks, _, _ = predictor.predict()


    # Return mask or results to the client
    return masks
