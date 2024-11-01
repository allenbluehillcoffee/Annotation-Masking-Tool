import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from PIL import Image, ImageDraw
import numpy as np

# Model paths
checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

# Initialize the SAM2 predictor
predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))

# Load and prepare the image
image_path = "testphoto.jpg"  # Replace with your image file path
image = Image.open(image_path).convert("RGB")
image_np = np.array(image)  # Convert to numpy array for SAM2 compatibility

# Check for CUDA availability
device = "cuda" if torch.cuda.is_available() else "cpu"

# Run SAM2 prediction
with torch.inference_mode():
    with torch.autocast(device, dtype=torch.bfloat16):
        predictor.set_image(image_np)
        masks, _, _ = predictor.predict()  # Get the masks list

# Retrieve and process the first mask
if masks:
    mask = masks[0].astype(np.uint8) * 255  # Assuming mask is binary and needs scaling to 0-255
    mask_image = Image.fromarray(mask).convert("L")  # Convert mask to grayscale

    # Create an overlay by blending the mask onto the original image
    overlay = Image.new("RGBA", image.size)
    overlay.paste((255, 0, 0, 100), (0, 0), mask_image)  # Red overlay with transparency

    # Convert original image to RGBA to match overlay
    image_with_mask = image.convert("RGBA")
    image_with_mask = Image.alpha_composite(image_with_mask, overlay)

    # Display or save the image with the mask overlay
    image_with_mask.show()  # Display the image with the mask overlay
    image_with_mask.save("output_with_mask_overlay.png")  # Save to a file

    print("Mask overlay applied and saved as output_with_mask_overlay.png")
else:
    print("No masks were returned by the SAM2 predictor.")
