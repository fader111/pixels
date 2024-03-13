import cv2
import numpy as np
from misc import *
import yaml

# Загрузка конфигурации из YAML-файла
with open('config.yaml') as file:
    config = yaml.safe_load(file)

img_size =              config['pic_size']
color_mode =            config['color_mode']
quanted_colors_num =    config['quanted_colors_num']
pixel_size =            config['pixel_size']
custom_colors = config['custom_colors_large'] if color_mode else \
                config['custom_colors_grey'] 


def on_trackbar_change(value):
    pixel_size = cv2.getTrackbarPos('Pixel Size', 'Pixelizer')
    pixel_size = pixel_size if pixel_size > 3 else 3
    pixelized = pixeller(img, custom_colors, pixel_size=pixel_size)
    _,_ = get_pixels_num(img_size, pixel_size)
    cv2.imshow('Pixelizer', pixelized)


pic_folder = "C:\\Users\\5010858\\Downloads\\Telegram Desktop\\"
path_l = [
    "ola2.jpg",
    "ola.jpg",
    "photo_2024-03-01_20-53-25.jpg", 
    "ola2-removebg-preview.png",
    "photo_2021-10-08_16-10-06.jpg",
    "IMG_20230723_140643.jpg", # 5
    "2_5226645942744125784.jpg",  # с гирляндой
    "ataranov.jpg"  # 7
]
path_ = pic_folder + path_l[5]
# path_ = r"C:\Users\Anton\Pictures\IMG_20210908_163738.png"
# path_ = r"C:\Users\Anton\Pictures\lpr1.png"
path_ = r"C:\Users\Anton\Projects\nextblog2\public\images\man_in_yard.jpg"

if __name__ == '__main__':

    if color_mode:
        img = cv2.imread(path_)
    else:
        img = cv2.imread(path_,0)
        img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
    
    img = resizeAndPad(img, img_size)
    
    # квантуем цвета методом k-means
    # cv2.imshow('before_quant', img)
    img = color_quantization(img, quanted_colors_num)
    cv2.imshow('ater_quant', img)

    pixelized = pixeller(img, custom_colors, pixel_size=pixel_size)

    cv2.namedWindow('Pixelizer')
    cv2.createTrackbar('Pixel Size', 'Pixelizer',
                       pixel_size, 12, on_trackbar_change)

    cv2.imshow('Pixelizer', pixelized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
