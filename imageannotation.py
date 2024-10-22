import json
import tornado.ioloop
import tornado.web
from tornado.web import RequestHandler, MainHandler, UploadHandler
import runsam2


class AnnotateHandler(RequestHandler):
    def post(self):
        data = json.loads(self.request.body)
        bounding_box = data  # The bounding box is received 
        print(f"Received bounding box: {bounding_box}")
        
        # Here you would pass the bounding box to SAM2 model
        # For simplicity, we will assume you have a function called run_sam2()

        result = runsam2.run_sam2(bounding_box)
        self.write({"status": "Bounding box processed", "result": result})



def make_app():
    return tornado.web.Application([
        (r"/", MainHandler),
        (r"/upload", UploadHandler),
        (r"/annotate", AnnotateHandler),
        (r"/static/(.*)", tornado.web.StaticFileHandler, {"path": "uploads"}),  # Serve static files
    ])
