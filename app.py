import os
import tornado.ioloop
import tornado.web
import json
import runsam2 

class MainHandler(tornado.web.RequestHandler):
    def get(self):
        self.render("index.html")

class UploadHandler(tornado.web.RequestHandler):
    def post(self):
        if not os.path.exists("uploads"):
            os.makedirs("uploads")
        
        files = self.request.files['file']
        file_paths = []
        for f in files:
            output_file = os.path.join("uploads", f.filename)
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

        # Load existing data or create new structure
        if os.path.exists(save_path):
            with open(save_path, "r") as f:
                existing_data = json.load(f)
        else:
            existing_data = {}

        # Update or add annotations for the image
        existing_data[file_name] = {
            "points": annotation.get("points", []),
            "boundingBoxes": annotation.get("boundingBoxes", [])
        }

        # Write the updated data back to the JSON file
        with open(save_path, "w") as f:
            json.dump(existing_data, f, indent=4)

        self.write({"status": "success"})

def make_app():
    return tornado.web.Application([
        (r"/", MainHandler),
        (r"/upload", UploadHandler),
        (r"/save_annotations", SaveAnnotationsHandler),  # New save handler for annotations
        (r"/static/(.*)", tornado.web.StaticFileHandler, {"path": "uploads"}),  # Serve static files
    ])

if __name__ == "__main__":
    app = make_app()
    app.listen(8888)
    print("Server started at http://localhost:8888")
    tornado.ioloop.IOLoop.current().start()
