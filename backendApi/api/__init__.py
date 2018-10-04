from flask_restplus import Api

from config import API_TITLE, API_VERSION, API_DESC
from .index import api as index

api = Api(
	title=API_TITLE,
	version=API_VERSION,
	description=API_DESC)

api.namespaces.clear()
api.add_namespace(index)
