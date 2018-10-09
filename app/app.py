import collections
import json
import logging
import mimetypes
import os

import signal
import time
import threading
import uuid

import requests

from tornado import escape, httpserver, ioloop, web
from tornado.options import define, options, parse_command_line

try:
    import Queue as queue
except ImportError:
    import queue

define("port", default=1688, help="the port of app server")
define("model_server", default="http://127.0.0.1:5200", help="the ImageCaption Model Server")

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"), format='[%(asctime)s] %(levelname)s: %(message)s', datefmt='%m/%d/%y %H:%M:%S')

static_img_path = "public/static/contents/"
img_prefix = "App_"
image_captions = collections.OrderedDict()
VALID_EXT = ['.png', '.jpg', '.jpeg']
error_raised = []
app_cookie = 'ICR' + str(uuid.uuid4())

"""
RequestHandler definition
"""

class BaseHandler(web.RequestHandler):
    def set_default_header(self):
        self.set_header('Access-Control-Allow-Origin', '*')
        self.set_header('Access-Control-Allow-Credentials', 'true')
        self.set_header('Access-Control-Allow-Headers', 'x-requested-with')
        self.set_header('Access-Control-Allow-Methods', 'POST, GET, PUT, DELETE')

    def get_current_user(self):
        return escape.to_basestring(self.get_secure_cookie(app_cookie))


class LoginHandler(BaseHandler):
    """Set a cookie for user when start in Application"""
    def post(self):
        if not self.get_secure_cookie(app_cookie):
            user_id = str(uuid.uuid4())
            self.set_secure_cookie(app_cookie, user_id)
            logging.info('New user cookie:' + app_cookie)
        else:
            logging.info('User cookie: ' + self.current_user)


class MainHandler(BaseHandler):
    """The default Handler"""
    def get(self):
        result={}
        clean_up_old_images()
        image_caption_list = get_image_captions(self.current_user)

        result['cookie_key'] = app_cookie
        image_captions= []
        i = 1
        for k,v in  image_caption_list.items():
            temp= {}
            k= k.replace('public','')
            temp['id']= i
            temp['pic']= k
            temp['captions']= v
            image_captions.append(temp)
            i+=1
        result['data'] = image_captions

        self.finish(json.dumps( result))


class DetailHandler(BaseHandler):
    def get(self):
        result={}
        user_image_captions = get_image_captions(self.current_user)
        image = self.get_argument('image', None)
        if not image:
            self.set_status(400)
            return self.finish("400: Missing image")
        if image not in user_image_captions:
            self.set_status(404)
            return self.finish("404: Image not found")
        predictions = user_image_captions[image]
        result['predictions'] = predictions

        self.finish(json.dumps( result))


class UploadHandler(BaseHandler):
    @web.authenticated
    def post(self):
        try:
            requests.get(model_server)
        except requests.exceptions.ConnectionError:
            logging.error("Lost connection to the ImageCaption Model Server at " + options.model_server)
            self.send_error(404)
            return

        finish_ret = []
        threads = []
        ret_queue = queue.Queue()
        user_img_prefix = get_user_img_prefix(self.current_user)

        new_files = self.request.files['file']
        for file_des in new_files:
            file_name = user_img_prefix + file_des['filename']

            if valid_file_ext(file_name):
                rel_path = static_img_path + file_name
                with open(rel_path, 'wb') as output_file:
                    output_file.write(file_des['body'])
                t = threading.Thread(target=get_captions_queued,
                                     args=(rel_path, ret_queue))
                threads.append(t)
                t.start()

        for t in threads:
            t.join()

        sorted_ret = sorted(list(ret_queue.queue), key=lambda t: t[0].lower())
        for rel_path, caption in sorted_ret:
            finish_ret.append({
                "file_name": rel_path,
                "caption": caption[0]['caption']
            })

        if not finish_ret:
            self.send_error(400)
            return
        sort_image_captions()

        temp= {}
        img= rel_path.replace('public','')
        temp['pic']= img
        temp['captions']= caption

        self.finish(json.dumps(temp))

class CleanupHandler(BaseHandler):
    @web.authenticated
    def delete(self):
        clean_up_user_images(self.current_user)

"""
common function definition
"""

def valid_file_ext(filename):
    _filename, file_extension = os.path.splitext(filename)
    valid = file_extension.lower() in VALID_EXT
    if not valid:
        logging.warning('Invalid file extension: ' + file_extension)
    return valid

def get_user_img_prefix(user_id):
    user_id = user_id if user_id else ""
    return img_prefix + user_id + "-"


def valid_user_img(user_id, img):
    """Checks if the given user uploaded the given image"""
    default_img = not img.startswith(static_img_path + img_prefix)
    user_img = img.startswith(static_img_path + get_user_img_prefix(user_id))
    current_user_img = user_img if user_id else False
    return default_img or current_user_img


def get_image_captions(user_id):
    return collections.OrderedDict(
        (k, v) for k, v in image_captions.items() if valid_user_img(user_id, k)
    )


def get_captions_queued(img_path, ret_queue):
    caption = get_captions(img_path)
    ret_queue.put((img_path, caption))


def get_captions(img_path):
    """Get captions from imagecaption model server"""
    mime_type = mimetypes.guess_type(img_path)[0]
    with open(img_path, 'rb') as img_file:
        file_form = {'image': (img_path, img_file, mime_type)}
        r = requests.post(url=model_server, files=file_form)
    try:
        r.raise_for_status()
    except requests.exceptions.HTTPError as e:
        error_raised.append(e)
        raise
    cap_json = r.json()
    caption = cap_json['predictions']
    image_captions[img_path] = caption
    return caption


def sort_image_captions():
    global image_captions
    image_captions = collections.OrderedDict(
        sorted(image_captions.items(), key=lambda t: t[0].lower()))

def getall_samples():
    """Get all sample's images captions"""
    threads = []
    rel_img_list = get_image_list()
    for img in rel_img_list:
        t = threading.Thread(target=get_captions, args=(img,))
        threads.append(t)
        t.start()
    for t in threads:
        t.join()

    sort_image_captions()

def get_image_list():
    """Gets list of images with relative paths from static dir"""
    image_list = sorted(os.listdir(static_img_path))
    rel_img_list = [static_img_path + s for s in image_list]
    return rel_img_list


def clean_up_user_images(user_id=None):
    """Cleans up user's images"""
    img_prefixs = get_user_img_prefix(user_id) if user_id else img_prefix
    img_list = get_image_list()
    for img_file in img_list:
        if img_file.startswith(static_img_path + img_prefixs):
            os.remove(img_file)
            image_captions.pop(img_file)


def clean_up_old_images():
    """Cleans up old images"""
    img_list = get_image_list()
    exp_time = time.time() - (24 * 60 * 60)
    for img_file in img_list:
        if (img_file.startswith(static_img_path + img_prefix)
                and os.stat(img_file).st_ctime < exp_time):
            os.remove(img_file)
            image_captions.pop(img_file)
            logging.info("Deleted expired image: " + img_file)


def signal_handler(sig, frame):
    ioloop.IOLoop.current().add_callback_from_signal(shutdown)


def shutdown():
    clean_up_user_images()
    server.stop()
    logging.info("Stopping ImageCaptioning server")
    ioloop.IOLoop.current().stop()

"""
Route definition
"""

def init_app():
    handlers = [
        (r"/", MainHandler),
        (r"/index", MainHandler),
        (r"/upload", UploadHandler),
        (r"/cleanup", CleanupHandler),
        (r"/detail", DetailHandler),
        (r"/login", LoginHandler)
    ]

    configs = {
        'static_path': 'contents',
        "cookie_secret": os.urandom(32),
        "debug": True
    }

    return web.Application(handlers, **configs)


def main():
    parse_command_line()

    global model_server
    model_server = options.model_server

    if '/model/predict' not in options.model_server:
        model_server = options.model_server.rstrip('/') + "/model/predict"

    logging.info("Connecting to ImageCaption Model Server at %s", model_server)

    try:
        requests.get(model_server)
    except requests.exceptions.ConnectionError:
        logging.error("Cannot connect to the ImageCaption Model Server at " + options.model_server)
        raise SystemExit

    logging.info("Starting Image captioning server")
    app = init_app()

    global server
    server = httpserver.HTTPServer(app)
    server.listen(options.port)
    signal.signal(signal.SIGINT, signal_handler)
    getall_samples()

    logging.info("Use Ctrl+C to stop Image captioning server")
    ioloop.IOLoop.current().start()


if __name__ == "__main__":
    main()
