import os
import tornado.ioloop
import tornado.web
import torch
from PIL import Image
import numpy as np
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# Directory to save and serve images
SAVE_DIR = "./saved_images"
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# SAM2 setup (initialize model once)
checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))

class MaskImageHandler(tornado.web.RequestHandler):
    async def post(self):
        # Retrieve the image and bounding box coordinates from the form
        fileinfo = self.request.files['image'][0]
        filename = fileinfo['filename']
        file_body = fileinfo['body']

        # Bounding box coordinates sent from the front-end (multiple boxes in JSON-like format)
        bbox_coords_list = self.get_argument('bbox_coords').split(';')  # Expected format: 'x1,y1,x2,y2;x1,y1,x2,y2;...'
        bboxes = [list(map(int, bbox.split(','))) for bbox in bbox_coords_list]

        # Save the original uploaded image
        input_image_path = os.path.join(SAVE_DIR, filename)
        with open(input_image_path, 'wb') as f:
            f.write(file_body)

        # Open the image using PIL
        image = Image.open(input_image_path).convert("RGB")
        image_np = np.array(image)

        print(f"Image dimensions: {image_np.shape}")
        print(f"Bounding box coordinates: {bboxes}")

        # Create a mask overlay for the entire image (same dimensions)
        final_mask_overlay = np.zeros_like(image_np)

        # Assuming image_np is the original image
        image_height, image_width = image_np.shape[:2]

        # Loop through each bounding box
        for bbox in bboxes:
            x_min, y_min, x_max, y_max = bbox

            # Ensure bounding box is within image dimensions
            if x_min < 0 or y_min < 0 or x_max > image_width or y_max > image_height:
                print(f"Skipping invalid bounding box: {x_min, y_min, x_max, y_max}")
                continue

            # Crop the image using the bounding box
            cropped_image_np = image_np[y_min:y_max, x_min:x_max]

            # Run SAM2 to get masks only for the cropped region
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                predictor.set_image(cropped_image_np)
                masks, _, _ = predictor.predict()

            # Ensure mask size matches the cropped region size
            mask_resized = np.zeros((y_max - y_min, x_max - x_min, 3), dtype=np.uint8)

            # Map the mask back into the original-sized image
            for mask in masks:
                mask_resized[mask == 1] = [255, 0, 0]  # Red color for the mask

            # Overlay the mask on the original image at the correct coordinates
            final_mask_overlay[y_min:y_max, x_min:x_max] = mask_resized

        # Convert the final mask overlay to a PIL image and blend with original
        final_mask_overlay_image = Image.fromarray(final_mask_overlay)

        # Blend the original image with the mask overlay
        blended_image = Image.blend(Image.fromarray(image_np), final_mask_overlay_image, alpha=0.5)

        # Save the final blended image
        output_image_filename = f"masked_{filename}"
        output_image_path = os.path.join(SAVE_DIR, output_image_filename)
        blended_image.save(output_image_path)

        # Redirect to display the image
        self.redirect(f'/display?file={output_image_filename}')


class DisplayImagesHandler(tornado.web.RequestHandler):
    def get(self):
        file = self.get_argument('file')

        # HTML to display the masked image and provide a "Save Image" button
        self.write(f"""
            <html>
            <head>
                <title>Masked Image</title>
                <style>
                    body {{
                        font-family: Arial, sans-serif;
                        text-align: center;
                        background-color: #f4f4f4;
                        margin: 0;
                        padding: 20px;
                    }}
                    h2 {{
                        color: #333;
                    }}
                    img {{
                        width: 400px;
                        height: auto;
                        margin: 10px;
                        border: 2px solid #ccc;
                        border-radius: 5px;
                        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
                    }}
                    .upload-link {{
                        display: inline-block;
                        margin-top: 20px;
                        padding: 10px 15px;
                        background-color: #007BFF;
                        color: white;
                        text-decoration: none;
                        border-radius: 5px;
                    }}
                    .upload-link:hover {{
                        background-color: #0056b3;
                    }}
                    .save-link {{
                        display: inline-block;
                        margin-top: 20px;
                        padding: 10px 15px;
                        background-color: #28a745;
                        color: white;
                        text-decoration: none;
                        border-radius: 5px;
                    }}
                    .save-link:hover {{
                        background-color: #218838;
                    }}
                </style>
            </head>
            <body>
            <h2>Masked Image</h2>
            <img src="/static/{file}" alt="Masked Image" />
            <br><a href="/" class="upload-link">Upload Another Image</a>
            </body>
            </html>
        """)

class UploadImageHandler(tornado.web.RequestHandler):
    def get(self):
        # HTML form for uploading and drawing bounding boxes
        self.write("""
            <html>
            <head>
                <title>Upload Image</title>
                <style>
                    body {{
                        font-family: Arial, sans-serif;
                        text-align: center;
                        background-color: #f4f4f4;
                        margin: 0;
                        padding: 20px;
                    }}
                    h2 {{
                        color: #333;
                    }}
                    .center {{
                        display: flex;
                        justify-content: center;
                        align-items: center;
                        flex-direction: column;
                    }}
                    input[type="file"], input[type="submit"] {{
                        margin-top: 10px;
                        padding: 10px 20px;
                        border-radius: 5px;
                        cursor: pointer;
                    }}
                    canvas {{
                        border: 2px solid #ccc;
                        border-radius: 5px;
                    }}
                    .upload-btn {{
                        background-color: #007BFF;
                        color: white;
                        border: none;
                        border-radius: 25px;
                        font-size: 16px;
                        cursor: pointer;
                        padding: 15px;
                        transition: background-color 0.3s;
                    }}
                    .upload-btn:hover {{
                        background-color: #0056b3;
                    }}
                </style>
            </head>
            <body>
            <h2>Upload Image and Draw Bounding Boxes</h2>
            <div class="center">
                <form id="imageForm" action="/mask" method="post" enctype="multipart/form-data">
                    <input type="file" id="imageInput" name="image" accept="image/*" required />
                    <canvas id="imageCanvas" width="400" height="300"></canvas>
                    <input type="hidden" id="bboxCoords" name="bbox_coords" />
                    <input type="submit" class="upload-btn" value="Mask Image" />
                </form>
            </div>

            <script>
                const imageInput = document.getElementById('imageInput');
                const imageCanvas = document.getElementById('imageCanvas');
                const bboxCoordsInput = document.getElementById('bboxCoords');
                const ctx = imageCanvas.getContext('2d');
                let image = new Image();
                let startX, startY, endX, endY;
                let boundingBoxes = [];

                // Load image into canvas when file is selected
                imageInput.onchange = function(event) {{
                    const file = event.target.files[0];
                    const reader = new FileReader();
                    reader.onload = function(e) {{
                        image.src = e.target.result;
                        image.onload = function() {{
                            // Set canvas size to match image dimensions
                            imageCanvas.width = image.width;
                            imageCanvas.height = image.height;
                            ctx.clearRect(0, 0, imageCanvas.width, imageCanvas.height);
                            ctx.drawImage(image, 0, 0, imageCanvas.width, imageCanvas.height);
                        }};
                    }};
                    reader.readAsDataURL(file);
                }};

                // Restrict drawing within image boundaries
                imageCanvas.onmousedown = function(e) {{
                    startX = Math.max(0, Math.min(e.offsetX, imageCanvas.width));  // Clamp to image width
                    startY = Math.max(0, Math.min(e.offsetY, imageCanvas.height)); // Clamp to image height
                }};

                // Handle click to add polygon points
                imageCanvas.onclick = function(e) {{
                    const x = Math.max(0, Math.min(e.offsetX, imageCanvas.width));  // Clamp to image width
                    const y = Math.max(0, Math.min(e.offsetY, imageCanvas.height)); // Clamp to image height

                    points.push({{x: x, y: y}});
                    
                    // Draw point and connect lines between points
                    ctx.fillStyle = 'blue';
                    ctx.beginPath();
                    ctx.arc(x, y, 3, 0, 2 * Math.PI);
                    ctx.fill();
                    
                    if (points.length > 1) {{
                        ctx.strokeStyle = 'blue';
                        ctx.lineWidth = 2;
                        ctx.beginPath();
                        ctx.moveTo(points[points.length - 2].x, points[points.length - 2].y);
                        ctx.lineTo(points[points.length - 1].x, points[points.length - 1].y);
                        ctx.stroke();
                    }}
                }};

                // Handle space key press to close the polygon
                window.onkeydown = function(e) {{
                    if (e.key === ' ') {{
                        if (points.length > 2) {{
                            // Close the polygon by connecting the last point to the first
                            ctx.strokeStyle = 'red';
                            ctx.lineWidth = 2;
                            ctx.beginPath();
                            ctx.moveTo(points[points.length - 1].x, points[points.length - 1].y);
                            ctx.lineTo(points[0].x, points[0].y);
                            ctx.stroke();

                            // Save the polygon points as a string and send it to the server
                            const polygonCoords = points.map(p => `${{p.x}},${{p.y}}`).join(';');
                            polygonCoordsInput.value = polygonCoords;

                            // Prevent form submission while space is pressed
                            e.preventDefault();
                        }}
                    }}
                }};
            </script>
            </body>
            </html>
        """)

    def post(self):
        # Get the uploaded image
        uploaded_file = self.request.files['image'][0]
        image = Image.open(io.BytesIO(uploaded_file['body']))

        # Get the polygon coordinates from the form
        polygon_coords = self.get_argument("polygon_coords")

        # Process the polygon coordinates
        mask_array = self.process_polygon(image, polygon_coords)

        # Save the mask or do further processing here
        # For example, you might want to save the mask as an image
        mask_image = Image.fromarray(mask_array * 255)  # Convert to image with 255 for mask
        mask_image.save("output_mask.png")  # Save mask image

        # Respond back to the user (you can change this as needed)
        self.write("Image uploaded and mask created successfully!")

    def process_polygon(self, image, polygon_coords):
        """
        Process the polygon coordinates and generate a mask.
        """
        # Convert the polygon coordinates string into a list of (x, y) tuples
        points = [tuple(map(int, point.split(','))) for point in polygon_coords.split(';')]
        
        # Create a mask image the same size as the original image
        mask = Image.new('L', image.size, 0)
        draw = ImageDraw.Draw(mask)
        
        # Draw the polygon on the mask
        draw.polygon(points, outline=1, fill=1)
        
        # Convert the mask to a numpy array and return
        mask_array = np.array(mask)
        return mask_array

def make_app():
    return tornado.web.Application([
        (r"/", UploadImageHandler),          # Endpoint for the upload form and drawing bounding boxes
        (r"/mask", MaskImageHandler),        # Endpoint for processing and masking with bounding box
        (r"/display", DisplayImagesHandler),  # Endpoint to display masked image
        (r"/static/(.*)", tornado.web.StaticFileHandler, {"path": SAVE_DIR}),  # Static file handler for images
    ])

if __name__ == "__main__":
    app = make_app()
    app.listen(8888)
    print("Server started at http://localhost:8888")
    tornado.ioloop.IOLoop.current().start()
