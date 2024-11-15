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
CROPPED_DIR = "cropped_images"
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)
if not os.path.exists(BOX_DIR):
    os.makedirs(BOX_DIR)
if not os.path.exists(CROPPED_DIR):
    os.makedirs(CROPPED_DIR)

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



            self.write({"status": "success", "message": "Annotations saved on server"})
            
            await self.maskimages()
        except Exception as e:
            self.write({"status": "error", "message": f"An error occurred: {str(e)}"})
            print("Error in SaveHandler:", str(e))
            

    async def maskimages(self):
        box_path = "./box_points"
        image_path = './uploads'
        save_path = "./masked_outputs"
        cropped_save_path = "./cropped_images"

        os.makedirs(save_path, exist_ok=True)

        image_files = os.listdir(image_path)
        box_paths = os.listdir(box_path)

        with open("./box_points/annotations.json", "r") as file:
            data = json.load(file)

        for individual_image in image_files:
            image_data_path = "/static/" + individual_image

            if image_data_path in data["annotations"]:
                bounding_box = data["annotations"][image_data_path]["boundingBoxes"]
                print(bounding_box)
            else:
                continue

            image = Image.open("./uploads/" + individual_image).convert("RGB")
            original_width, original_height = image.size

            new_height = (original_height/original_width) * 600

            scale_x = original_width / 600
            scale_y = original_height / new_height

            for index, box in enumerate(bounding_box):
                left = box['minX'] * scale_x
                upper = box['minY'] * scale_y
                right = left + box['width'] * scale_x
                lower = upper + box['height'] * scale_y

                padding_x = (right - left) * 0.1
                padding_y = (lower - upper) * 0.1
                padded_left = max(0, left - padding_x)
                padded_upper = max(0, upper - padding_y)
                padded_right = min(original_width, right + padding_x)
                padded_lower = min(original_height, lower + padding_y)

                # Crop and resize the padded area to 512x512
                padded_crop = image.crop((padded_left, padded_upper, padded_right, padded_lower))
                padded_crop_resized = padded_crop.resize((512, 512),  Image.LANCZOS)

                cropped_image_name = f"cropped_{index}_{individual_image}"
                padded_crop_resized.save(os.path.join(cropped_save_path, cropped_image_name))

                cropped_image = image.crop((left, upper, right, lower))
                cropped_image_np = np.array(cropped_image)

                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                    predictor.set_image(cropped_image_np)
                    masks, _, _ = predictor.predict()

                mask_overlay = np.zeros_like(cropped_image_np)

                for mask in masks:
                    mask_overlay[mask == 1] = [255, 255, 255]

                full_overlay = Image.new("RGB", image.size, (0, 0, 0))

                mask_overlay_image = Image.fromarray(mask_overlay)
                full_overlay.paste(mask_overlay_image, (int(left), int(upper)))

                # Blend the original image with the mask overlay (no labels or bounding boxes)
                blended_image = Image.blend(image, full_overlay, alpha=1)

                # Save the image with masks
                blended_image_name = f"masked_{index}_{individual_image}"
                blended_image.save(os.path.join(save_path, blended_image_name))





def make_app():
    return tornado.web.Application([
        (r"/", MainHandler),
        (r"/upload", UploadHandler),
        (r"/save_annotations", SaveAnnotationsHandler),
        (r"/static/(.*)", tornado.web.StaticFileHandler, {"path": "uploads"}),
    ], max_body_size=2000 * 1024 * 1024) 

if __name__ == "__main__":
    app = make_app()
    app.listen(8888)
    print("Server started at http://localhost:8888")
    tornado.ioloop.IOLoop.current().start()
