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

window_title = f'Pixelizer colors{quanted_colors_num}' 

def show_pic():
    # cv2.imshow('Pixelizer', pixelized)
    cv2.imshow('Pixelizer', np.hstack((img_res, img_quant_color, pixelized)))

def on_trackbar_change(value):
    pixel_size = cv2.getTrackbarPos('Pixel Size', window_title)
    quanted_colors_num = cv2.getTrackbarPos('Colors', window_title)
    
    pixel_size = pixel_size if pixel_size > 3 else 3

    img_quant_color = color_quantization(img_res, quanted_colors_num)
    quanted_colors_num = quanted_colors_num if quanted_colors_num > 3 else 3
    pixelized = pixeller(img_quant_color, custom_colors, pixel_size=pixel_size)
    _,_ = get_pixels_num(img_size, pixel_size)
    cv2.imshow(window_title, np.hstack((img_res, img_quant_color, pixelized)))


pic_folder = "C:\\Users\\5010858\\Downloads\\Telegram Desktop\\"
orig_num = 55
path_l = [
    "ola2.jpg", #0
    "ola.jpg",  #1
    "photo_2024-03-01_20-53-25.jpg", #2
    "ola2-removebg-preview.png",    #3
    "photo_2021-10-08_16-10-06.jpg",
    "IMG_20230723_140643.jpg", # 5
    f"orig{orig_num}.webp",  # 6
    "ataranov.jpg", # 7
    "Nasta1.png",   # 8
    "ola_tasa.jpg",  # 9
    "nada1.png", # 10
    "sandwich.png", # 11
]
path_ = pic_folder + path_l[11]
# path_ = r"C:\Users\Anton\Pictures\IMG_20210908_163738.png"
# path_ = r"C:\Users\Anton\Pictures\lpr1.png"
# path_ = r"C:\Users\Anton\Projects\nextblog2\public\images\man_in_yard.jpg"

if __name__ == '__main__':

    if color_mode:
        img_orig = cv2.imread(path_)
    else:
        img_orig = cv2.imread(path_,0)
        img_orig = cv2.cvtColor(img_orig, cv2.COLOR_GRAY2RGB)
    
    img_res = resizeAndPad(img_orig, img_size)
    
    # квантуем цвета методом k-means
    # cv2.imshow('before_quant', img)
    img_quant_color = color_quantization(img_res, quanted_colors_num)
    # cv2.imshow('ater_quant', img_quant_color)

    pixelized = pixeller(img_quant_color, custom_colors, pixel_size=pixel_size)

    cv2.namedWindow(window_title)
    cv2.createTrackbar('Pixel Size', window_title,
                       pixel_size, 12, on_trackbar_change)
    cv2.createTrackbar('Colors', window_title, quanted_colors_num, 50, on_trackbar_change)
                       

    cv2.imshow(window_title, np.hstack((img_res, img_quant_color, pixelized)))
    # show_pic()
    cv2.waitKey(0)
    cv2.destroyAllWindows()
