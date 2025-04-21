import cv2 as cv
import os
import json

from computer_vision import detect_points

INPUT_DIR = "in1"
files = [f"{file}" for file in os.listdir(INPUT_DIR) if file.endswith(".jpg")]
final_json = dict()

for file in files:
    name = file.split(".")[0]
    img = cv.imread(f'{INPUT_DIR}/{file}')

    points = detect_points(img, name)

    # Записываем метки
    metka_index = name.split("_")[0] + '_' + name.split("_")[1]
    local_index = name.split("_")[2]

    if metka_index not in final_json:
        final_json[metka_index] = dict()

    final_json[metka_index][local_index] = points

    print(file)

json.dump(final_json, open("data.json", "w"))
