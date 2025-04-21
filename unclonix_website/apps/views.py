from flask import render_template, redirect, request
from random import randint

import os
import cv2 as cv
from .computer_vision import detect_points
from .hash_functions import bishop_function

path = './apps/static/mediafiles/'
os.makedirs(path, exist_ok=True)

def views(app):
    @app.errorhandler(404)
    def not_found(e):
        return redirect("/")

    @app.route('/', methods=['GET'])
    def index():
        return render_template('main.html')

    @app.route('/sitemap.xml', methods=['GET'])
    def Sitemap():
        return redirect('Sitemap.xml')

    @app.route('/api/get_hash', methods=['POST'])
    def get_hash():
        try:
            file = list(request.files.values())[0]

            file_name = f'{str(randint(1000000, 100000000000000))}.{file.filename.split(".")[-1]}'
            file.save(f'{path}/{file_name}')

            img = cv.imread(f'{path}/{file_name}')
            points, points_all = detect_points(img)
            hash_val = bishop_function(points)

            print(f'Hash val: {hash_val}')

            # 18 метка
            return str(hash_val == "4dff530f460aff98")
        except Exception as e:
            print(e)
            return "error"