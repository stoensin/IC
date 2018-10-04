DEBUG = False

# Flask-restplus settings
RESTPLUS_MASK_SWAGGER = False
SWAGGER_UI_DOC_EXPANSION = 'none'

API_TITLE = 'ImageCaption Model Server API'
API_DESC = 'API for ImageCaption Model'
API_VERSION = '0.1'

MODEL_NAME = 'im2txt'
DEFAULT_MODEL_PATH = 'assets/{}'.format(MODEL_NAME)
