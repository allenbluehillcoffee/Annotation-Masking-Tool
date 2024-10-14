import os
import tornado.ioloop
import tornado.web

class MainHandler(tornado.web.RequestHandler):
    def get(self):
        self.render("index.html")

class UploadHandler(tornado.web.RequestHandler):
    def post(self):
        files = self.request.files['file']
        for f in files:
            # Save uploaded files
            output_file = os.path.join("uploads", f.filename)
            with open(output_file, "wb") as f_out:
                f_out.write(f.body)
        self.write({"status": "File uploaded"})

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
