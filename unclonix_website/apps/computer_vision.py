import numpy as np
import cv2 as cv
import skimage as ski

WIDTH = 1000
START_MAX_LWN = 1160


def detect_points(img, name=""):
    # Меняем разрешение изображения на стандартное для нас
    if img.shape[1] >= img.shape[0]:
        dim = (START_MAX_LWN, int(START_MAX_LWN * img.shape[0] / img.shape[1]))
    else:
        dim = (int(START_MAX_LWN * img.shape[1] / img.shape[0]), START_MAX_LWN)

    img = cv.resize(img, dim, interpolation=cv.INTER_LANCZOS4)
    cv.imwrite(f'./apps/static/mediafiles/resized.jpg', img)

    # Получаем маску внутри которой есть нужная нам информация
    hsv_full_mask_blur, hsv_full_mask = get_mask(img, name)  # hsv_full_mask содержит внутри себя прорехи

    # Находим координаты окружности метки
    (x, y, r), is_ok = get_circle(hsv_full_mask_blur, img, name)

    # Аффинное преобразование изображения, чтобы сдвинуть круг в центр нового изображения
    affined_mask_blur, affined_mask, affined_img = affine_transform([hsv_full_mask_blur, hsv_full_mask, img], (x, y, r))

    # После преобразования маска перестала быть бинарной (появилась "серая" граница), поэтому избавляемся от нее, округляя
    affined_mask_blur = np.round(affined_mask_blur / 255).astype(np.uint8) * 255
    affined_mask = np.round(affined_mask / 255).astype(np.uint8) * 255

    # Применяем маску, т.е. убираем фон метки, делая его черным
    affined_img = cv.bitwise_and(affined_img, affined_img, mask=affined_mask_blur)

    # Убираем из цветного изображения медианный цвет метки, и опять к получившемуся изображению применяем маску
    affined_img_minus_median = remove_median_color(affined_img, affined_mask, name)
    affined_img_minus_median = cv.bitwise_and(affined_img_minus_median, affined_img_minus_median, mask=affined_mask_blur)

    if name != "":
        cv.imwrite(f'otladka/cv/{name}-0_img.jpg', img)
        # cv.imwrite(f'otladka/cv/{name}-4_mask.png', affined_mask_blur)
        # cv.imwrite(f'otladka/cv/{name}-5_invert.png', affined_img)
        cv.imwrite(f'otladka/cv/{name}-6_affined_img.jpg', affined_img)
        cv.imwrite(f'otladka/cv/{name}-7_affined_img_minus.jpg', affined_img_minus_median)

    # Берем среднее (преобразовываем в серый)
    gray = cv.cvtColor(affined_img_minus_median, cv.COLOR_BGR2GRAY)

    # TODO Уменьшение с 15 до 5 решило в основном проблемы, ушло переполнение и блестки более различимы, но все равно граница создает проблемы
    # TODO Нормализовывать надо тоже локально
    # TODO Попробовать аналог softmax
    # Нормализуем по яркости
    gray_normalize = get_normalize(gray, name, 8)

    # Немного размываем для устранения шумов (главным образом границы, если она мешает)
    # gray_normalize_blur = cv.medianBlur(gray_normalize, 11)
    gray_normalize_blur = gray_normalize

    if name != "":
        cv.imwrite(f'otladka/cv/{name}-10_gray.jpg', gray)
        cv.imwrite(f'otladka/cv/{name}-11_gray_blur.jpg', get_normalize(cv.GaussianBlur(gray, (31, 31), 0), name, 8))
        cv.imwrite(f'otladka/cv/{name}-12_gray_normalize.jpg', gray_normalize)
        # cv.imwrite(f'otladka/cv/{name}-12_gray_normalize_minus.jpg', gray_normalize - conv_gray)
        # cv.imwrite(f'otladka/cv/{name}-13_gray_normalize_blur.jpg', gray_normalize_blur)

    # Ищем блестки
    points = get_points(gray_normalize_blur, affined_img, name)

    # Создаем список с элементами Points c характеристиками блесток
    points_all = points_characteristics(points, gray_normalize_blur, affined_mask_blur)

    # Фильтруем их
    points = filter_points(points_all, affined_img, "")

    return points, [p.data() for p in points_all]


def get_mask(bgr_img, name=""):
    # Меняем цветовое пространство на hsv
    hsv = cv.cvtColor(bgr_img, cv.COLOR_BGR2HSV)

    # В нем делаем маску по принципу принадлежности к области (примерно оранжевый круг)
    lower = np.array([0, 100, 127], dtype="uint8")
    upper = np.array([255, 255, 255], dtype="uint8")
    hsv_mask = cv.inRange(hsv, lower, upper)

    # Маска, которая позволит исключить лепестки (возможно имеет смысл adaptiveThreshold)
    # ret, petal_mask = cv.threshold(cv.split(hsv)[2], 0, 255, cv.THRESH_OTSU)

    # Объединяем маски (если оба пикселя в каждой маске верны), чтобы получить весьма точную маску
    # hsv_full_mask = cv.bitwise_and(hsv_mask, petal_mask)
    hsv_full_mask = hsv_mask

    # Вроде и так норм работает, но возможно потом придется добавить (или нет)
    # kernel = np.ones((3, 3), np.uint8)
    # closing = cv.morphologyEx(hsv_mask, cv.MORPH_OPEN, kernel)  # Или MORTH_CLOSE

    # Чем больше параметр, ksize - тем более ровный кргу получается (>70 - почти идеальный круг)
    hsv_full_mask_blur = cv.medianBlur(hsv_full_mask, 81)

    # hsv_gray_masked_blured = cv.GaussianBlur(hsv_mask, (15, 15), 0)

    # Для отладки
    if name != "":
        # cv.imwrite(f'otladka/cv/{name}-1_hsv.jpg', hsv)
        # for i in range(hsv.shape[0]):
        #     for j in range(hsv.shape[1]):
        #         hsv[i, j] = [0, hsv[i, j][1], hsv[i, j][2]]
        # cv.imwrite(f'otladka/cv/{name}-1_hsv2.jpg', cv.cvtColor(hsv, cv.COLOR_HSV2BGR))
        # cv.imwrite(f'otladka/cv/{name}-2_hsv_mask.jpg', hsv_mask)
        # cv.imwrite(f'otladka/cv/{name}-3_petal_mask.jpg', petal_mask)
        # cv.imwrite(f'otladka/cv/{name}-2_hsv_full_mask.jpg', hsv_full_mask)
        # cv.imwrite(f'otladka/cv/{name}-3_hsv_full_mask_blur.jpg', hsv_full_mask_blur)
        pass
        # cv.imwrite(f'otladka/cv/000_{name}.jpg', hsv_full_mask)

    return hsv_full_mask_blur, hsv_full_mask


def get_circle(mask, image_add_circle=None, name=""):
    # Круги
    # param2 - чем меньше, тем больше кругов, param1 как будто не влияет
    circles = cv.HoughCircles(mask, cv.HOUGH_GRADIENT, 1, 1000, minRadius=100, maxRadius=500, param1=10, param2=10)

    # ensure at least some circles were found
    if circles is None:
        return (0, 0, 0), False

    # convert the (x, y) coordinates and radius of the circles to integers
    circles = np.round(circles[0, :]).astype("int")

    # loop over the (x, y) coordinates and radius of the circles
    x, y, r = circles[0]

    return (x, y, r), True


def affine_transform(img_list, coords, name=""):
    x, y, r = coords

    from_points = np.array([[x, y], [x + r, y], [x, y + r]]).astype(np.float32)
    to_point = np.array([[500, 500], [990, 500], [500, 990]]).astype(np.float32)

    warp_mat = cv.getAffineTransform(from_points, to_point)

    new_img_list = []
    for i, img in enumerate(img_list):
        new_img = cv.warpAffine(img, warp_mat, (1000, 1000))
        new_img_list.append(new_img)

    # if name != "":
    #     for i, img in enumerate(new_img_list):
    #         cv.imwrite(f'otladka/cv/affined_{i}_{name}.jpg', img)

    return new_img_list


def remove_median_color(img, mask, name=""):
    # Считаем медианный цвет без учета черного (поля)
    img = img.astype(np.int16)
    img_splited_flatten = [i.flatten() for i in cv.split(img)]
    median_color = np.array([np.median(i[i != 0]) for i in img_splited_flatten])
    # median_color = median_color.astype(np.uint8)

    # affined_img_orange - остается только оранжевый цвет, причем где раньше был черный (дырки и вокруг), там теперь медианный цвет
    median_color_broadcasted = np.broadcast_to(median_color, img.shape)
    affined_img_orange = (
        cv.bitwise_and(img, img, mask=mask) +
        cv.bitwise_and(median_color_broadcasted, median_color_broadcasted, mask=(255 - mask))
    ).astype(np.uint8)  # uint8 - чтобы работал cv.medianBlur
    affined_img_orange_blur = cv.medianBlur(affined_img_orange, 21)
    
    if name != "":
        cv.imwrite(f'otladka/cv/{name}-4_affined_img_orange.png', affined_img_orange)
        cv.imwrite(f'otladka/cv/{name}-4_affined_img_orange_blur.png', affined_img_orange_blur)
        pass

    img_minus_median = np.abs(img - affined_img_orange_blur.astype(np.int16)).astype(np.uint8)

    return img_minus_median


def get_normalize(gray, name, mean_brightness_will_be=15):
    gray_gblur = cv.GaussianBlur(gray, (31, 31), 0)
    cv.imwrite(f'otladka/cv/{name}-11_gray_blur.jpg', gray_gblur)

    mean = np.mean(gray)
    multiply_coef = mean_brightness_will_be / mean
    return np.minimum(multiply_coef * gray.astype(np.float64), 255).astype("uint8")


def get_points(img, image_add_circle2, name):
    image_add_circle2 = image_add_circle2.copy()
    image_add_circle = cv.cvtColor(img.copy(), cv.COLOR_GRAY2RGB)

    # TODO несколько таких с разными сигмами допустимыми и разными threshold
    blobs_log = ski.feature.blob_log(img, min_sigma=WIDTH / 1000, max_sigma=WIDTH / 120, num_sigma=5, threshold=.18, overlap=0.7)
    blobs_log[:, 2] = (blobs_log[:, 2] * np.sqrt(2)) + 1

    points = []
    for blob in blobs_log:
        x, y, r = int(blob[1].item()), int(blob[0].item()), int(blob[2].item())
        points.append((x, y, r))

        cv.circle(image_add_circle, (x, y), r + 2, (152, 251, 152), 1)
        cv.circle(image_add_circle2, (x, y), r + 3, (152, 251, 152), 1)
    cv.circle(image_add_circle, (500, 500), 490, (127, 0, 127), 1)

    cv.imwrite(f'otladka/cv/{name}-20_circles_log.jpg', image_add_circle)
    # cv.imwrite(f'otladka/cv/yy_{name}.jpg', image_add_circle)
    cv.imwrite(f'otladka/cv/zz_{name}.jpg', image_add_circle2)

    return points


class Point:
    def __init__(self, x, y, r, point_coef=0, strong=0):
        self.x = x
        self.y = y
        self.r = r
        self.point_coef = point_coef
        self.strong = strong

    def data(self):
        return self.x, self.y, (self.r, round(self.point_coef, 2), int(self.strong))


def points_characteristics(points, gray, mask_blur):
    # Фильтруем по сумме значений в gray, чтобы в дальнейшем иметь возможность различать точки на границе
    new_points = list()

    for x, y, r in points:
        # Маска (ядро) для данной точки на основе параметров окружности
        kernel = np.zeros_like(mask_blur)
        cv.circle(kernel, (x, y), r, (255, 255, 255), -1)

        # Считаем коэффициент нахождения внутри круга (0 - полностью за кругом, 1 - полностью внутри)
        max_point_sum = np.sum(kernel)
        point_sum = np.sum(cv.bitwise_and(kernel, mask_blur))
        point_coef =  point_sum / max_point_sum

        # Считаем "мощность точки"
        strong_coef = np.sum(cv.bitwise_and(kernel, gray)) / 255

        new_points.append(Point(x, y, r, point_coef, strong_coef))

    return new_points


def array_to_points(arr):
    return [Point(i[0], i[1], i[2][0], i[2][1], i[2][2]) for i in arr]


def filter_points(points, img, name=""):
    # Фильтруем по сумме значений в gray и Убираем точки на границе
    new_points = list()

    for p in points:
        # Формируем изображение, которое в дальнейшем станет отладочным
        cv.circle(img, (p.x, p.y), p.r, (0, 255, 0), 1)
        cv.putText(img, f"{round(p.strong)};{round(p.point_coef, 2)};{p.r}", (p.x + p.r + 5, p.y + 2), cv.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 0), 1, cv.LINE_AA)

        # Тут костыль для метки 18
        if (p.point_coef > 0.999 and p.r >= 6 and p.strong >= 85) or (p.point_coef > 0.98 and p.r >= 8 and p.strong >= 136):  # Если точка достаточно лежит внутри
            new_points.append(p)

    points = new_points

    # Сдвоенные точки превращаем в одну точку
    new_points = list()
    dont_take_points = list()
    for i, p1 in enumerate(points):
        if p1 in dont_take_points:  # Если точка уже обработана (она пересекается с другой и их среднее уже записано)
            continue

        for p2 in points[i + 1:]:
            if (p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2 < (p1.r + p2.r) ** 2:  # Если точки p1 и p2 пересекаются
                new_points.append(
                    Point(
                        (p1.x + p2.x) // 2,
                        (p1.y + p2.y) // 2,
                        p1.r + p2.r,
                        min(p1.point_coef, p2.point_coef),
                        min(p1.strong, p2.strong)
                    )
                )  # x, y, r - средние и сумма для r
                dont_take_points.append(p2)
                break
        else:
            new_points.append(p1)
    points = new_points

    if name != "":
        points_img = np.zeros((1000, 1000, 3), np.uint8)
        points_len = len(points)

        font = cv.FONT_HERSHEY_SIMPLEX
        font_size = 1
        font_thickness = 2
        text = f'{points_len} points'
        cv.putText(points_img, text, (10, 30), font, font_size, (255, 255, 255), font_thickness, cv.LINE_AA)
        # cv.putText(img, text, (10, 30), font, font_size, (255, 255, 255), font_thickness, cv.LINE_AA)

        for p in points:
            cv.circle(points_img, (p.x, p.y), p.r, (255, 255, 255), 1)
            cv.putText(points_img, f"{round(p.strong)};{p.r}", (p.x + p.r + 5, p.y + 2), font, 0.3, (255, 255, 0), 1, cv.LINE_AA)

        # for p in points:
        #     cv.circle(img, (p.x, p.y), p.r, (255, 255, 255), 2)
        #     cv.putText(img, f"{round(p.strong)};{p.r}", (p.x + p.r + 5, p.y + 2), font, 0.3, (255, 255, 0), 1, cv.LINE_AA)

        cv.imwrite(f'otladka/cv/yy_{name}.jpg', points_img)
        cv.imwrite(f'otladka/cv/further_{name}.jpg', img)
        cv.imwrite(f'otladka/cv/{name}-25_points.jpg', points_img)

    return [p.data() for p in points]
