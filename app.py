import os
import json
import tornado.ioloop
import tornado.web
import numpy as np
from PIL import Image
import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# SAM2 model setup
checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))

# Define paths
UPLOAD_DIR = "uploads"
SAVE_DIR = "masked_outputs"
BOX_DIR = "box_points"
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)
if not os.path.exists(BOX_DIR):
    os.makedirs(BOX_DIR)

class MainHandler(tornado.web.RequestHandler):
    def get(self):
        self.render("index.html")

class UploadHandler(tornado.web.RequestHandler):
    def post(self):
        files = self.request.files['file']
        file_paths = []
        for f in files:
            output_file = os.path.join(UPLOAD_DIR, f.filename)
            with open(output_file, "wb") as f_out:
                f_out.write(f.body)
            file_paths.append(f"/static/{f.filename}")
        self.write({"status": "File uploaded", "file_paths": file_paths})

class SaveAnnotationsHandler(tornado.web.RequestHandler):
    async def post(self):
        try:
            data = json.loads(self.request.body)
            annotations = data  

            if not annotations:
                self.write({"status": "error", "message": "No annotations provided"})
                return

            # Define the save location for annotations
            annotations_file_path = os.path.join(BOX_DIR, "annotations.json")

            # Save annotations to the server
            with open(annotations_file_path, "w") as f:
                json.dump(annotations, f, indent=4)

            # Call the method to mask images based on the saved annotations
            output_images = await self.mask_images(annotations)

            self.write({"status": "success", "message": "Annotations saved on server", "output_images": output_images})
        except Exception as e:
            self.write({"status": "error", "message": f"An error occurred: {str(e)}"})
            print("Error in SaveHandler:", str(e))  # Log error for troubleshooting

    async def mask_images(self, annotations):
        output_images = []  # Store paths to output images

        # Process each image in the annotations
        for image_path, annotation_data in annotations.items():
            if 'boundingBoxes' not in annotation_data:
                continue
                
            image_path = image_path.replace('/static/', '/uploads/')
            bounding_boxes = annotation_data.get("boundingBoxes", [])
            for bbox in bounding_boxes:
                min_x = bbox['minX']
                min_y = bbox['minY']
                max_x = min_x + bbox['width']
                max_y = min_y + bbox['height']

                # Call the masking function for the bounding box
                output_image_path = await self.mask_single_image(image_path, (min_x, min_y, max_x, max_y))
                output_images.append(output_image_path)

        return output_images

    async def mask_single_image(self, image_path, bbox_coords):
        input_image_path = os.path.join(UPLOAD_DIR, image_path.lstrip("/"))  # Remove leading slash for path
        if not os.path.exists(input_image_path):
            return {"status": "error", "message": f"File not found: {input_image_path}"}

        image = Image.open(input_image_path).convert("RGB")
        image_np = np.array(image)
        x_min, y_min, x_max, y_max = bbox_coords

        cropped_image_np = image_np[y_min:y_max, x_min:x_max]
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            predictor.set_image(cropped_image_np)
            masks, _, _ = predictor.predict()

        mask_resized = np.zeros((y_max - y_min, x_max - x_min, 3), dtype=np.uint8)
        for mask in masks:
            mask_resized[mask == 1] = [255, 0, 0]

        final_mask_overlay = np.zeros_like(image_np)
        final_mask_overlay[y_min:y_max, x_min:x_max] = mask_resized

        final_mask_overlay_image = Image.fromarray(final_mask_overlay)
        blended_image = Image.blend(Image.fromarray(image_np), final_mask_overlay_image, alpha=0.5)
        output_image_filename = f"masked_{os.path.basename(image_path)}"
        output_image_path = os.path.join(SAVE_DIR, output_image_filename)
        blended_image.save(output_image_path)

        return f"/static/{output_image_filename}"





def make_app():
    return tornado.web.Application([
        (r"/", MainHandler),
        (r"/upload", UploadHandler),
        (r"/save_annotations", SaveAnnotationsHandler),
        (r"/static/(.*)", tornado.web.StaticFileHandler, {"path": "uploads"}),
    ])

if __name__ == "__main__":
    app = make_app()
    app.listen(8888)
    print("Server started at http://localhost:8888")
    tornado.ioloop.IOLoop.current().start()
