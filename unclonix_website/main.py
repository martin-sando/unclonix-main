#!/usr/bin/python3
import os

from apps import app

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=2103, debug=os.environ.get('DEBUG'))
