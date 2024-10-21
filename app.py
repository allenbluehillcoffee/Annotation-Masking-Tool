import os
import tornado.ioloop
import tornado.web

class MainHandler(tornado.web.RequestHandler):
    def get(self):
        self.render("index.html")

class UploadHandler(tornado.web.RequestHandler):
    def post(self):
        if not os.path.exists("uploads"):
            os.makedirs("uploads")  # Create uploads directory if it doesn't exist
        
        files = self.request.files['file']
        file_paths = []  # To store paths of uploaded files
        for f in files:
            output_file = os.path.join("uploads", f.filename)
            with open(output_file, "wb") as f_out:
                f_out.write(f.body)
            # Append the file path to the list
            file_paths.append(f"/static/{f.filename}")
        
        # Return the file paths
        self.write({"status": "File uploaded", "file_paths": file_paths})

def make_app():
    return tornado.web.Application([
        (r"/", MainHandler),
        (r"/upload", UploadHandler),
        (r"/static/(.*)", tornado.web.StaticFileHandler, {"path": "uploads"}),  # Serve static files
    ])  

if __name__ == "__main__":
    app = make_app()
    app.listen(8888)
    print("Server started at http://localhost:8888")
    tornado.ioloop.IOLoop.current().start()
