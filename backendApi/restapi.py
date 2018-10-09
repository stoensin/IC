import os
from flask import Flask
from api import api

app = Flask(__name__)

app.config.from_object('config')

if 'APP_CONFIG' in os.environ:
	app.config.from_envvar('APP_CONFIG')
api.init_app(app)

if __name__ == '__main__':
	app.run(host='127.0.0.1')
