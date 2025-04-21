import flask
from flask_compress import Compress
import os

from .views import views

app = flask.Flask(__name__, static_url_path='/', static_folder='static')
Compress(app)

app.config['SECRET_KEY'] = 'asiovdfnuhgarihg'

app.jinja_env.globals.update(os=os)
app.jinja_env.globals.update(app=app)
app.jinja_env.globals.update(len=len)
app.jinja_env.globals.update(str=str)
app.jinja_env.globals.update(enumerate=enumerate)
app.jinja_env.globals.update(ver='0.0')

views(app)
