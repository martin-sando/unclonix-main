#!/usr/bin/python3
import os

from apps import app

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3016, debug=os.environ.get('DEBUG'))
