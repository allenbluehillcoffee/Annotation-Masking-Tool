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

# New handler to save annotated files and annotations
class SaveHandler(tornado.web.RequestHandler):
    def post(self):
        data = tornado.escape.json_decode(self.request.body)
        annotations = data.get('annotations', {})  # Get the annotations dictionary
        save_dir = "annotated"

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Save annotations to files
        for file_name, points in annotations.items():
            annotation_file = os.path.join(save_dir, f"{os.path.basename(file_name)}_annotations.json")
            with open(annotation_file, "w") as af:
                json.dump(points, af)  # Save the points as a JSON file
                print(f"Saved annotations for {file_name}: {annotation_file}")

        self.write({"status": "Annotations saved", "saved_annotations": list(annotations.keys())})

# class AnnotationHandler(tornado.web.RequestHandler):ddf 


def make_app():
    return tornado.web.Application([
        (r"/", MainHandler),
        (r"/upload", UploadHandler),
        (r"/save_annotations", SaveHandler),  # New save handler for annotations
        (r"/static/(.*)", tornado.web.StaticFileHandler, {"path": "uploads"}),  # Serve static files
    ])

if __name__ == "__main__":
    app = make_app()
    app.listen(8888)
    print("Server started at http://localhost:8888")
    tornado.ioloop.IOLoop.current().start()
