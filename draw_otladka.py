from math import sin, cos, pi
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

r = 8
font = cv.FONT_HERSHEY_SIMPLEX


class Point:
    def __init__(self, x, y, id):
        self.x = x
        self.y = y
        self.id = id

    def r(self):
        return (self.x ** 2 + self.y ** 2) ** 0.5

    def set_zero_point(self, x_0, y_0):
        self.x = self.x - x_0
        self.y = self.y - y_0

def circumcircle_center(p1: Point, p2: Point, p3: Point):
    """
    Вычисляет координаты центра описанной окружности по трём точкам.
    Аргументы:
        p1, p2, p3 - объекты класса Point с атрибутами x и y
    Возвращает:
        Кортеж (x, y) - координаты центра окружности, или None, если точки коллинеарны
    """
    # Извлекаем координаты точек
    x1, y1 = p1.x, p1.y
    x2, y2 = p2.x, p2.y
    x3, y3 = p3.x, p3.y

    # Проверяем, не лежат ли точки на одной прямой
    # Если определитель равен 0, точки коллинеарны
    d = 2 * (x1*(y2 - y3) + x2*(y3 - y1) + x3*(y1 - y2))
    if d == 0:
        return None  # Точки лежат на одной прямой, окружность невозможна

    # Вычисляем координаты центра (x, y) по формуле
    # Используем метод пересечения серединных перпендикуляров
    ux = ((x1**2 + y1**2)*(y2 - y3) + (x2**2 + y2**2)*(y3 - y1) + (x3**2 + y3**2)*(y1 - y2)) / d
    uy = ((x1**2 + y1**2)*(x3 - x2) + (x2**2 + y2**2)*(x1 - x3) + (x3**2 + y3**2)*(x2 - x1)) / d

    return (int(ux), int(uy))

def get_area_radius_center(p1, p2, p3):
    """
    Вычисляет радиус описанной окружности по координатам трёх точек.
    Аргументы: координаты точек (x1,y1), (x2,y2), (x3,y3)
    Возвращает: радиус окружности или (0, 0), если точки лежат на одной прямой
    """
    # Вычисляем длины сторон треугольника
    a = int(((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2) ** 0.5)  # расстояние между точками 1 и 2
    b = int(((p2.x - p3.x) ** 2 + (p2.y - p3.y) ** 2) ** 0.5)  # расстояние между точками 2 и 3
    c = int(((p3.x - p1.x) ** 2 + (p3.y - p1.y) ** 2) ** 0.5)  # расстояние между точками 3 и 1

    # Используем формулу площади треугольника через координаты
    area = abs((p1.x * (p2.y - p3.y) + p2.x * (p3.y - p1.y) + p3.x * (p1.y - p2.y)) / 2)

    # Проверяем, не лежат ли точки на одной прямой
    if area == 0:
        return 0, 0  # Точки коллинеарны

    # Вычисляем радиус по формуле: R = abc / (4K), где K - площадь
    radius = (a * b * c) / (4 * area)
    return int(area), int(radius), circumcircle_center(p1, p2, p3)


def get_coords(coords, angle):
    x = cos(angle) * (coords[0] - 500) + sin(angle) * (coords[1] - 500)
    y = -sin(angle) * (coords[0] - 500) + cos(angle) * (coords[1] - 500)
    return int(x) + 500, int(y) + 500


def draw_img(points_xy, triangles, adjacency_matrix, vertex_order, hash_hex, filename, image):
    img_matrix = cv.resize(adjacency_matrix.astype(np.uint8), dsize=(1000, 1000), interpolation=cv.INTER_NEAREST_EXACT).astype(bool)
    plt.imsave(f"otladka/00_{filename}.png", img_matrix, cmap=plt.get_cmap("gray"))

    # im_vector_graph = np.zeros((1000, 1000, 3), np.uint8)

    vertex_order = vertex_order.tolist()

    points = [Point(x=points_xy[i][0], y=points_xy[i][1], id=vertex_order.index(i)) for i in range(len(points_xy))]

    points_sorted = sorted(points, key=lambda p: p.id)

    # Поиск опорной точки
    anchor_point = points_sorted[-1]

    # Нормализуем так, чтобы максимальная длина была 500
    rotate_angle = np.angle((anchor_point.x - 500) + (anchor_point.y - 500) * 1j, deg=False) + pi / 2

    M = cv.getRotationMatrix2D((500, 500), 180 * rotate_angle / pi, 1.0)
    im_vector_graph = cv.warpAffine(image, M, (1000, 1000))

    for triangle in triangles:
        point0 = points[triangle[0]]
        point1 = points[triangle[1]]
        point2 = points[triangle[2]]

        area, radius, circle_center = get_area_radius_center(point0, point1, point2)

        x0, y0 = get_coords((point0.x, point0.y), rotate_angle)
        x1, y1 = get_coords((point1.x, point1.y), rotate_angle)
        x2, y2 = get_coords((point2.x, point2.y), rotate_angle)

        circle_center = get_coords(circle_center, rotate_angle)

        # Рисуем линии графа
        cv.line(im_vector_graph, (x0, y0), (x1, y1), (255, 0, 0), 1)
        cv.line(im_vector_graph, (x0, y0), (x2, y2), (255, 0, 0), 1)
        cv.line(im_vector_graph, (x1, y1), (x2, y2), (255, 0, 0), 1)

        # Центр треугольника
        center_x, center_y = get_coords(((point0.x + point1.x + point2.x) // 3, (point0.y + point1.y + point2.y) // 3), rotate_angle)

        cv.circle(im_vector_graph, (center_x, center_y), 1, (0, 0, 255), 2)
        cv.circle(im_vector_graph, circle_center, radius, (255,45,200), 1)
        cv.circle(im_vector_graph, circle_center, 3, (255,255,255), 1)
        cv.putText(im_vector_graph, f"{area} | {radius}", (circle_center[0] + 3, circle_center[1] + 3), font, 0.4, (0, 0, 255), 1, cv.LINE_AA)


    for i, point in enumerate(points):
        x, y = get_coords((point.x, point.y), rotate_angle)
        cv.circle(im_vector_graph, (x, y), r - 2, (255, 255, 255), 1)
        cv.putText(im_vector_graph, f"#{point.id}|{int(points_xy[i][2][2])}|{round(points_xy[i][2][1], 3)}", (x + r, y), font, 0.4, (0, 255, 255), 1, cv.LINE_AA)

    cv.putText(im_vector_graph, f"{len(points)} points", (2, 25), font, 1, (255, 255, 255), 2, cv.LINE_AA)
    cv.putText(im_vector_graph, hash_hex, (2, 995), font, 0.75, (255, 255, 255), 2, cv.LINE_AA)

    cv.imwrite(f"otladka/{filename}.jpg", im_vector_graph)
