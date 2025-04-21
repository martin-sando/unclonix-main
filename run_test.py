import json
import os
import cv2 as cv
from tqdm import tqdm
from time import sleep, time
from natsort import os_sorted

from computer_vision import detect_points, filter_points, array_to_points
from hash_functions import bishop_function

need_otladka = True
filename = "data2.json"

INPUT_DIR = "input_resize"


def statistics(hashes):
    set_list = set(hashes)

    amount_dict = dict()
    for hash in set_list:
        amount_dict[hash] = hashes.count(hash)

    # Sort by values
    amount_dict = {k: v for k, v in sorted(amount_dict.items(), key=lambda item: item[1])}

    max_amount_of_one_hash = list(amount_dict.values())[-1]
    hashes_len = len(hashes)
    return max_amount_of_one_hash, hashes_len


def run_test(test_data, hash_func, filename=None):
    hashes = list()

    for d in os_sorted(test_data):
        img = cv.imread(f'otladka/cv/{filename}_{d}-6_affined_img.jpg')
        points = filter_points(array_to_points(test_data[d]), img, f"{filename}_{d}")
        hash_val = hash_func(points, filename=f"{filename}_{d}" if need_otladka else None, image=img)
        hashes.append(hash_val)
        print(f"│ {hash_val}    {filename}_{d}")

    max_amount_of_one_hash, hashes_len = statistics(hashes)
    is_ok = (100 * max_amount_of_one_hash / hashes_len) >= 95
    print(f'└─── Распознано {max_amount_of_one_hash}/{hashes_len} ~ {int((100 * max_amount_of_one_hash / hashes_len))}%. {"Все хорошо ✅" if is_ok else "Плохо ❌"}')
    return max_amount_of_one_hash, hashes_len, is_ok


if __name__ == "__main__":
    files = [f"{file}" for file in os.listdir(INPUT_DIR) if file.lower().endswith(".jpg")]
    final_json = dict()

    print("Началось распознавание меток...")
    sleep(0.1)
    start = time()

    # Распознование меток
    for file in tqdm(files, ncols=100):
       name = file.split(".")[0]
       img = cv.imread(f'{INPUT_DIR}/{file}')

       points, points_all = detect_points(img, name if need_otladka else "")

       # Записываем метки
       metka_index = name.split("_")[0] + '_' + name.split("_")[1]
       local_index = name.split("_")[2]

       if metka_index not in final_json:
           final_json[metka_index] = dict()

       final_json[metka_index][local_index] = points_all

    json.dump(final_json, open(filename, "w"))

    print(f"Распознавание окончено. 1 фото в среднем заняло {round((time() - start) / len(files), 2)}с\n")

    json_data = json.load(open(filename))
    #  json_data = final_json

    # Подсчет хэшей
    metkas_len = 0
    recognized_correctly = 0

    for key in json_data:
        metkas_len += 1
        print(f"┌─── Метка: {key}")
        max_amount_of_one_hash, hashes_len, is_ok = run_test(json_data[key], hash_func=bishop_function, filename=key)
        recognized_correctly += is_ok
        print(" ")

    print("\nИтог:")
    print(f"Распознано корректно меток: {recognized_correctly}/{metkas_len} ~ {100 * recognized_correctly // metkas_len}%")
