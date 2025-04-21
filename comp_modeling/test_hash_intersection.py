from random import randint
from tqdm import tqdm, trange
from hash_functions import bishop_function

BLOBS_NUM = [30, 50]

WIDTH = 1000
EXPERIMENTS_NUM = 1_000_000

all_hashes = set()
for i in trange(EXPERIMENTS_NUM, ncols=80):
    # points = [(randint(0, WIDTH), randint(0, WIDTH)) for p in range(randint(BLOBS_NUM[0], BLOBS_NUM[1]))]
    points = [(randint(0, WIDTH), randint(0, WIDTH)) for p in range(30)]
    h = bishop_function(points)
    all_hashes.add(h)

    if (i + 1) % 500_000 == 0:
        print(len(all_hashes), '/', i + 1)

print(len(all_hashes))