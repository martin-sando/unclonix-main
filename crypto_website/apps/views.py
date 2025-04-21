from flask import render_template, redirect, request
from user_agents import parse
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
        return 404

    @app.route('/', methods=['GET'])
    def index():
        # Получаем User-Agent из запроса
        user_agent_string = request.headers.get('User-Agent', '')
        user_agent = parse(user_agent_string)

        # Определяем тип устройства
        if user_agent.is_mobile or user_agent.is_tablet:
            return render_template('mobile.html')

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

            graph_filename = f"graph_{randint(1000000, 100000000000000)}"
            hash_val = bishop_function(points, filename=graph_filename)

            print(f'Hash val: {hash_val}')

            return {"hash": hash_val, "filename": graph_filename, "status": True}
        except Exception as e:
            print(e)
            return {"status": False}