from a2wsgi import ASGIMiddleware
from app import app

# This is the object PythonAnywhere's WSGI server will look for
application = ASGIMiddleware(app)
