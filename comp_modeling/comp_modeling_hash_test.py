from random import randint
from tqdm import tqdm, trange
from hash_functions import bishop_function

BLOBS_NUM = [30, 50]

WIDTH = 1000
COORDS_DELTA = 4
COORDS_DELTA_2 = COORDS_DELTA // 2

EXPERIMENTS_NUM = 10_000
EXPERIMENTS_PER_METKA = 20

good_metkas = 0
all_hashes = set()
hashes_num_per_metka = [0 for i in range(EXPERIMENTS_PER_METKA + 1)]

for i in trange(EXPERIMENTS_NUM, ncols=80):
    # TODO: генерить r и ф, а потом перевести в x и y
    # points_truth = [(randint(0, WIDTH), randint(0, WIDTH)) for p in range(30)]
    points_truth = [
        (randint(COORDS_DELTA_2, WIDTH - COORDS_DELTA_2), randint(COORDS_DELTA_2, WIDTH + COORDS_DELTA_2))
        for p in range(randint(BLOBS_NUM[0], BLOBS_NUM[1]))
    ]

    hashes_local = set()
    for j in range(EXPERIMENTS_PER_METKA):
        points = [[p[0] + randint(0, COORDS_DELTA) - COORDS_DELTA_2, p[1] + randint(0, COORDS_DELTA) - COORDS_DELTA_2] for p in points_truth]
        h = bishop_function(points)

        hashes_local.add(h)
        all_hashes.add(h)

    len_hashes_local = len(hashes_local)
    if len_hashes_local == 1:
        good_metkas += 1
    hashes_num_per_metka[len_hashes_local] += 1

    if (i + 1) % 500_000 == 0:
        print(len(all_hashes), '/', i + 1)

print(f"Всего хэшей в экспериментах получено: {len(all_hashes)}. При идеальном раскладе: {EXPERIMENTS_NUM}, при самом плохом - {EXPERIMENTS_PER_METKA * EXPERIMENTS_NUM}")
print(f"Хороших меток: {good_metkas}/{EXPERIMENTS_NUM} ~ {int(100 * good_metkas / EXPERIMENTS_NUM)}%")

print("hashes_num_per_metka:")
print(hashes_num_per_metka)
