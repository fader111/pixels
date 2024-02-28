import cv2
import numpy as np
# from config import *
from misc import *
import yaml

# Загрузка конфигурации из YAML-файла
with open('config.yaml') as file:
    config = yaml.safe_load(file)

pic_size_ = config['pic_size']
pic_width = config['pic_width']
pic_height = config['pic_height']
subst_colors_set = config['colors_set'] 
pixel_size = config['pixel_size']

def nearest_color_calc(orig_color, subst_color, subst_colors_set): 
    subst_color = orig_color
    return subst_color


def colors_fitting(blocks): 
    ''' берет блоки и заменяет цвета - в каждом блоке '''
    # for color in
    # cv2.imshow(blocks)
    # cv2.waitKey(0) 
    return blocks


def pixeller(img, pixel_size=5):
    pixel_size = pixel_size if pixel_size > 0 else 1
    height, width, channels = img.shape
    # Pad image
    pad_x = (pixel_size - width % pixel_size) % pixel_size
    pad_y = (pixel_size - height % pixel_size) % pixel_size
    img = np.pad(img, ((0, pad_y), (0, pad_x), (0, 0)), mode='reflect')
    # Reshape image into blocks and compute average color of each block
    h, w, c = img.shape
    blocks = np.mean(img.reshape(
        h//pixel_size, pixel_size, -1, pixel_size, c), axis=(1, 3))
    # Repeat average color of each block to fill corresponding region in the image
    # blocks[:,:,1] = 10
    # blocks[:,:,0] = 5
    # blocks[:,:,2] = 50
    blocks = colors_fitting(blocks)

    print (f"blocks len{len(blocks)} \nshape - {blocks.shape} \n blocks \n\n{blocks[:,:,:]}")

    output = np.repeat(np.repeat(blocks, pixel_size,
                       axis=1), pixel_size, axis=0)
    # Remove padding
    return output[:height, :width].astype("uint8")


def on_trackbar_change(value):
    pixel_size = cv2.getTrackbarPos('Pixel Size', 'Pixelizer')
    pixelized = pixeller(img, pixel_size=pixel_size)
    pixel_size = pixel_size if pixel_size > 0 else 1
    _,_ = get_pixels_num(img, pixel_size)
    cv2.imshow('Pixelizer', pixelized)


def get_pixels_num(img, pixel_size):
    pixels_num_w, pixels_num_h = round(pic_size_[0]/pixel_size), round(pic_size_[1]/pixel_size)
    print (f"pixels_num_w, pixels_num_h {pixels_num_w, pixels_num_h} tot number - {pixels_num_w * pixels_num_h}")
    return pixels_num_w, pixels_num_h


pic_folder = "C:\\Users\\5010858\\Downloads\\Telegram Desktop\\"
path_l = [
    "ola2.jpg",
    "ola.jpg",
    "ola2-removebg-preview.png",
    "photo_2021-10-08_16-10-06.jpg",
    "2_5226645942744125784.jpg",  # с гирляндой
    "ataranov.jpg"  # 5
]
path_ = pic_folder + path_l[2]
path_ = r"C:\Users\Anton\Pictures\IMG_20210908_163738.png"

if __name__ == '__main__':
    img = cv2.imread(path_)#,0)
    # img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
    img = resizeAndPad(img, pic_size_)
    # img = cv2.resize(img, (300, 300))
    # pixel_size = 50

    pixelized = pixeller(img, pixel_size=pixel_size)

    cv2.namedWindow('Pixelizer')
    cv2.createTrackbar('Pixel Size', 'Pixelizer',
                       pixel_size, 50, on_trackbar_change)

    cv2.imshow('Pixelizer', pixelized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
