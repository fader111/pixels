import cv2
import numpy as np
from misc import *
import yaml

def test_color_quantization1():
    img = np.array([
        [[255, 0, 0], [0, 255, 0], [0, 0, 255]],
        [[255, 255, 0], [255, 0, 255], [0, 255, 255]]
    ], np.uint8)

    k = 6
    result = color_quantization(img, k)
    unique_colors = np.unique(result.reshape(-1, result.shape[2]), axis=0)

    assert len(unique_colors) == k, f"Expected {k} colors, but found {len(unique_colors)} colors"

    print("Test test_color_quantization passed")

test_color_quantization1()


def test_color_quantization2():
    # Загрузка данных из файла config.yaml
    with open('config.yaml', 'r') as file:
        data = yaml.safe_load(file)

    # Создание палитры цветов
    colors_palette = data['custom_colors_large']

    # Создание изображения с тестовыми цветами
    img = np.zeros((len(colors_palette), 1, 3), np.uint8)
    for i, color in enumerate(colors_palette.values()):
        img[i] = color

    k = len(colors_palette)
    result = color_quantization(img, k)
    unique_colors = np.unique(result.reshape(-1, result.shape[2]), axis=0)

    assert len(unique_colors) == k, f"Expected {k} colors, but found {len(unique_colors)} colors"

    print("Test test_color_quantization2 passed")

test_color_quantization2()