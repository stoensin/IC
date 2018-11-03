import json
import logging
import os
import sys
import requests
import signal
import numpy

from tornado import escape, httpserver, ioloop, web
from tornado.options import define, options, parse_command_line

sys.path.append('../')
# from im2txt.server import ModelWrapper
from im2txt.densecap.server import ModelWrapper

# Command Line Options
define("port", default=5201, help="Port the web app will run on")


# Setup Logging
logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"),
                    format='[%(asctime)s] %(levelname)s: %(message)s',
                    datefmt='%m/%d/%y %H:%M:%S')

VALID_EXT = ['.png', '.jpg', '.jpeg']
error_raised = []

model_wrapper = ModelWrapper()

class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.integer):
            return int(obj)
        elif isinstance(obj, numpy.floating):
            return float(obj)
        elif isinstance(obj, numpy.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)

class BaseHandler(web.RequestHandler):
    def set_default_header(self):
        self.set_header('Access-Control-Allow-Origin', '*')
        self.set_header('Access-Control-Allow-Credentials', 'true')


class PredictHandler(BaseHandler):
     def get(self):
        result = {'status': 'connect success!'}
        self.finish(json.dumps(result,cls=MyEncoder))
    def post(self):
        result = {'status': 'error'}

        files = self.request.files

        image_data= files['image'][0]['body']
        preds = model_wrapper.predict(image_data)

        label_preds = [{'index': p[0], 'caption': p[1], 'probability': p[2]} for p in [x for x in preds]]
        result['predictions'] = label_preds
        result['status'] = 'ok'

        self.finish(json.dumps(result,cls=MyEncoder))


class MainHandler(BaseHandler):

     def get(self):
        result = {'status': 'connect success!'}
        self.finish(json.dumps(result,cls=MyEncoder))

def valid_file_ext(filename):
    """Checks if the given filename contains a valid extension"""
    _filename, file_extension = os.path.splitext(filename)
    valid = file_extension.lower() in VALID_EXT
    if not valid:
        logging.warning('Invalid file extension: ' + file_extension)
    return valid

def signal_handler(sig, frame):
    ioloop.IOLoop.current().add_callback_from_signal(shutdown)

def shutdown():
    logging.info("Stopping ImageCaption Model server")
    server.stop()
    ioloop.IOLoop.current().stop()


def make_app():
    handlers = [
        (r"/", MainHandler),
        (r"/model/predict", PredictHandler),
    ]

    configs = {
        "cookie_secret": os.urandom(32),
        "debug": True
    }

    return web.Application(handlers, **configs)


def main():
    parse_command_line()

    logging.info("Starting ImageCaption Model server")
    app = make_app()
    global server
    server = httpserver.HTTPServer(app)
    server.listen(options.port)
    signal.signal(signal.SIGINT, signal_handler)

    logging.info("Use Ctrl+C to stop ImageCaption Model server")
    ioloop.IOLoop.current().start()


if __name__ == "__main__":
    main()
