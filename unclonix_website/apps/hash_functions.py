import numpy as np
import hashlib
from scipy.spatial import Delaunay

from .draw_otladka import draw_img


def canonical_adj_matrix(adj_matrix):
    n = len(adj_matrix)
    if n != len(adj_matrix[0]):
        raise ValueError("Adjacency matrix must be square")

    adj_matrix = np.array(adj_matrix)

    # Степень точки
    degrees = np.sum(adj_matrix, axis=1)

    # Сумма степеней у прилегающих вершин, затем остальные и т.д.
    degrees_2 = np.sum(np.repeat(np.expand_dims(degrees, axis=0), n, axis=0), axis=1, where=adj_matrix.astype(bool))
    degrees_3 = np.sum(np.repeat(np.expand_dims(degrees_2, axis=0), n, axis=0), axis=1, where=adj_matrix.astype(bool))
    degrees_4 = np.sum(np.repeat(np.expand_dims(degrees_3, axis=0), n, axis=0), axis=1, where=adj_matrix.astype(bool))
    degrees_5 = np.sum(np.repeat(np.expand_dims(degrees_4, axis=0), n, axis=0), axis=1, where=adj_matrix.astype(bool))
    degrees_6 = np.sum(np.repeat(np.expand_dims(degrees_5, axis=0), n, axis=0), axis=1, where=adj_matrix.astype(bool))

    all_degrees = np.concatenate((
        np.expand_dims(degrees, axis=1),
        np.expand_dims(degrees_2, axis=1),
        np.expand_dims(degrees_3, axis=1),
        np.expand_dims(degrees_4, axis=1),
        np.expand_dims(degrees_5, axis=1),
        np.expand_dims(degrees_6, axis=1)
    ), axis=1)

    all_degrees = np.array(list(map(tuple, all_degrees)), dtype=[('a', int), ('b', int), ('c', int), ('d', int), ('e', int), ('f', int)])
    # print(np.sort(all_degrees, order=('a', 'b', 'c', 'd', 'e', 'f')))

    vertex_order = np.argsort(all_degrees, order=('a', 'b', 'c', 'd', 'e', 'f'))[::-1]

    canonical_matrix = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(n):
            canonical_matrix[i][j] = adj_matrix[vertex_order[i]][vertex_order[j]]

    return canonical_matrix, vertex_order


def extract_upper_triangular_flat(matrix):
    # Matrix + 1 - чтобы потом убирать 0, т.к. изначально состоит из 0 и 1
    upper_flat = np.triu(matrix + 1, k=1).ravel()
    result = upper_flat[upper_flat != 0] - 1
    return result


# Baranov Ivan's Solid Hash Of Points (BISHOP)
def bishop_function(points, filename=None, image=None):
    np_points = np.array([(p[0], p[1]) for p in points])
    tri = Delaunay(np_points)
    triangles = tri.simplices.tolist()

    n = len(points)

    # Создаем матрицу смежности
    adjacency_matrix = np.zeros((n, n), dtype=int)

    # Заполняем ее
    for triangle in triangles:
        adjacency_matrix[triangle[0], triangle[1]] = 1
        adjacency_matrix[triangle[1], triangle[2]] = 1
        adjacency_matrix[triangle[0], triangle[2]] = 1

        adjacency_matrix[triangle[1], triangle[0]] = 1
        adjacency_matrix[triangle[2], triangle[1]] = 1
        adjacency_matrix[triangle[2], triangle[0]] = 1

    # Приводим к каноническому виду
    adjacency_matrix, vertex_order = canonical_adj_matrix(adjacency_matrix)

    # Начинаем подсчет хэша для каждой лучшей точки
    hash_bin = "".join(map(str, extract_upper_triangular_flat(adjacency_matrix).tolist()))
    hash_hex = hashlib.shake_256(hash_bin.encode()).hexdigest(8)

    if filename is not None:  # Отладка
        draw_img(points, triangles, adjacency_matrix, vertex_order, hash_hex, filename, image)

    return hash_hex


if __name__ == "__main__":
    from time import time

    example_points = [
        [898, 585], [741, 349], [426, 38], [768, 116], [42, 490], [743, 207], [197, 557], [687, 426], [795, 640], [678, 94], [106, 429], [546, 947], [242, 213],
        [480, 319], [272, 511], [719, 379], [518, 861], [303, 856], [660, 785], [537, 240], [599, 297], [546, 328], [537, 675], [306, 605], [220, 131],
        [335, 185], [428, 601], [962, 579], [657, 332], [706, 351], [128, 331], [356, 671], [633, 305], [332, 111], [429, 562], [456, 394], [807, 494],
        [802, 805], [383, 107], [42, 448], [131, 229]
    ]

    st = time()
    bishop_function(example_points)
    print(time() - st)
