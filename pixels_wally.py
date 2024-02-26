import cv2, os
import numpy as np
import threading

def apply_filters(detail, brightness, contrast, color):
    # Применение эффектов пикселизации, изменения яркости, контрастности и цвета
    pixelized = cv2.resize(original_image, (detail, detail), interpolation=cv2.INTER_NEAREST)
    adjusted = cv2.addWeighted(pixelized, brightness, np.zeros_like(pixelized), 0, contrast)
    colored = cv2.applyColorMap(adjusted, color)
    return colored

def update_image():
    while True:
        cv2.imshow('Pixelizer', filtered_image)
        if cv2.waitKey(1) == ord('q'):
            break

def start_filter_thread():
    # Запуск обновления изображения в отдельном потоке
    threading.Thread(target=update_image).start()

def on_trackbar_change(value):
    # Callback-функция, вызываемая при изменении значений слайдеров
    detail = cv2.getTrackbarPos('Detail', 'Controls')
    brightness = cv2.getTrackbarPos('Brightness', 'Controls') / 100
    contrast = cv2.getTrackbarPos('Contrast', 'Controls') / 100
    color = cv2.getTrackbarPos('Color', 'Controls')

    global filtered_image
    filtered_image = apply_filters(detail, brightness, contrast, color)

# Считывание изображения с диска
image_path = 'path_to_your_image.jpg'
original_image = cv2.imread(image_path)
# Считывание изображения с диска
pic_folder ="C:\\Users\\5010858\\Downloads\\Telegram Desktop\\" 

path_l = [  "ola2.jpg",
            "ola.jpg",
            "ola2-removebg-preview.png",
            "photo_2021-10-08_16-10-06.jpg",
            "2_5226645942744125784.jpg", # с гирляндой
            "ataranov.jpg"                              # 5
        ]

image_path = os.path.join(pic_folder, path_l[0])
original_image = cv2.imread(image_path)

if __name__ == '__main__':
    cv2.namedWindow('Controls')
    cv2.createTrackbar('Detail', 'Controls', 1, 100, on_trackbar_change)
    cv2.createTrackbar('Brightness', 'Controls', 100, 200, on_trackbar_change)
    cv2.createTrackbar('Contrast', 'Controls', 100, 200, on_trackbar_change)
    cv2.createTrackbar('Color', 'Controls', 2, 14, on_trackbar_change)
    filtered_image = original_image.copy()

    # Создание окна с изображением
    cv2.namedWindow('Pixelizer')
    cv2.imshow('Pixelizer', filtered_image)

    # Запуск обновления изображения в отдельном потоке
    start_filter_thread()

    cv2.waitKey(0)
    cv2.destroyAllWindows()
