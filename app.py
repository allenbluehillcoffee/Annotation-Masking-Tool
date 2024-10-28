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
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

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
    def post(self):
        data = tornado.escape.json_decode(self.request.body)
        file_name = data.get("file_name")
        annotation = data.get("annotation")
        if not file_name or not annotation:
            self.write({"status": "error", "message": "Missing file name or annotation data"})
            return

        save_path = "data.json"
        if os.path.exists(save_path):
            with open(save_path, "r") as f:
                existing_data = json.load(f)
        else:
            existing_data = {}

        existing_data[file_name] = {
            "points": annotation.get("points", []),
            "boundingBoxes": annotation.get("boundingBoxes", [])
        }
        with open(save_path, "w") as f:
            json.dump(existing_data, f, indent=4)
        self.write({"status": "success"})

class MaskImageHandler(tornado.web.RequestHandler):
    async def post(self):
        filename = self.get_argument('filename')  # Read filename from form data
        bbox_coords_list = json.loads(self.get_argument('bbox_coords'))  # Read bounding box coordinates

        input_image_path = os.path.join(UPLOAD_DIR, filename)
        if not os.path.exists(input_image_path):
            self.write({"status": "error", "message": f"File not found: {input_image_path}"})
            return

        image = Image.open(input_image_path).convert("RGB")
        image_np = np.array(image)
        image_height, image_width = image_np.shape[:2]
        final_mask_overlay = np.zeros_like(image_np)

        for bbox in bbox_coords_list:
            x_min, y_min, x_max, y_max = bbox
            if x_min < 0 or y_min < 0 or x_max > image_width or y_max > image_height:
                print(f"Skipping invalid bounding box: {x_min, y_min, x_max, y_max}")
                continue
            cropped_image_np = image_np[y_min:y_max, x_min:x_max]
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                predictor.set_image(cropped_image_np)
                masks, _, _ = predictor.predict()

            mask_resized = np.zeros((y_max - y_min, x_max - x_min, 3), dtype=np.uint8)
            for mask in masks:
                mask_resized[mask == 1] = [255, 0, 0]
            final_mask_overlay[y_min:y_max, x_min:x_max] = mask_resized

        final_mask_overlay_image = Image.fromarray(final_mask_overlay)
        blended_image = Image.blend(Image.fromarray(image_np), final_mask_overlay_image, alpha=0.5)
        output_image_filename = f"masked_{filename}"
        output_image_path = os.path.join(SAVE_DIR, output_image_filename)
        blended_image.save(output_image_path)

        self.write({"status": "success", "output_image": f"/static/{output_image_filename}"})

def make_app():
    return tornado.web.Application([
        (r"/", MainHandler),
        (r"/upload", UploadHandler),
        (r"/save_annotations", SaveAnnotationsHandler),
        (r"/mask_image", MaskImageHandler),
        (r"/static/(.*)", tornado.web.StaticFileHandler, {"path": "uploads"}),
    ])

if __name__ == "__main__":
    app = make_app()
    app.listen(8888)
    print("Server started at http://localhost:8888")
    tornado.ioloop.IOLoop.current().start()
