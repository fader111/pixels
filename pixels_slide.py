import cv2
import numpy as np
from misc import *
import yaml

# Загрузка конфигурации из YAML-файла
with open('config.yaml') as file:
    config = yaml.safe_load(file)

pic_size_ = config['pic_size']
pic_width = config['pic_width']
pic_height = config['pic_height']
custom_colors = config['custom_colors_large'] 
pixel_size = config['pixel_size']


def nearest_color_calc(orig_color, custom_colors): 
    """ берем цвет из оригинальной картинки и вычисляем ближайший из палитры
        как расстояние от точки до точки в трехмерном пространстве
    """
    custom_colors_np = np.array(list(custom_colors.values()))
    distances = np.sqrt(np.sum((orig_color - custom_colors_np)**2,axis=1))
    # index_of_smallest = np.where(distances==np.amin(distances)) # косячит
    index_of_smallest = np.argmin(distances)
    subst_color = custom_colors_np[index_of_smallest]
    return subst_color


def colors_fitting(blocks, custom_colors): 
    ''' берет блоки и заменяет цвета - в каждом блоке '''
    out_blocks = np.zeros_like(blocks)
    for i in range(blocks.shape[0]):
        for j in range(blocks.shape[1]):  
            out_blocks[i, j] = nearest_color_calc(blocks[i, j], custom_colors)
            # out_blocks[i, j] = blocks[i, j]
    # cv2.imshow("blocks", blocks) 
    return out_blocks


def pixeller(img, pixel_size=5, min_pixel_size=3):
    pixel_size = pixel_size if pixel_size > min_pixel_size else min_pixel_size
    height, width, channels = img.shape
    # Pad image
    pad_x = (pixel_size - width % pixel_size) % pixel_size
    pad_y = (pixel_size - height % pixel_size) % pixel_size
    img = np.pad(img, ((0, pad_y), (0, pad_x), (0, 0)), mode='reflect')
    # Reshape image into blocks and compute average color of each block
    h, w, c = img.shape
    blocks = np.mean(img.reshape(
        h//pixel_size, pixel_size, -1, pixel_size, c), axis=(1, 3))
    blocks = colors_fitting(blocks, custom_colors)

    # print (f"blocks len{len(blocks)} \nshape - {blocks.shape} \n blocks \n\n{blocks[:,:,:]}")

    # Repeat average color of each block to fill corresponding region in the image
    output = np.repeat(np.repeat(blocks, pixel_size,
                       axis=1), pixel_size, axis=0)
    # Remove padding
    return output[:height, :width].astype("uint8")


def get_pixels_num(img, pixel_size):
    pixels_num_w, pixels_num_h = pic_size_[0]//pixel_size, pic_size_[1]//pixel_size
    print (f"w/h pixels {pixels_num_w, pixels_num_h} tot -{pixels_num_w * pixels_num_h} pxl size -{pixel_size}" )
    return pixels_num_w, pixels_num_h


def on_trackbar_change(value):
    pixel_size = cv2.getTrackbarPos('Pixel Size', 'Pixelizer')
    pixel_size = pixel_size if pixel_size > 3 else 3
    pixelized = pixeller(img, pixel_size=pixel_size)
    _,_ = get_pixels_num(img, pixel_size)
    cv2.imshow('Pixelizer', pixelized)


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
    color_mode = 1
    if color_mode:
        img = cv2.imread(path_)
    else:
        img = cv2.imread(path_,0)
        img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
    img = resizeAndPad(img, pic_size_)
    # img = cv2.resize(img, (300, 300))

    pixelized = pixeller(img, pixel_size=pixel_size)

    cv2.namedWindow('Pixelizer')
    cv2.createTrackbar('Pixel Size', 'Pixelizer',
                       pixel_size, 50, on_trackbar_change)

    cv2.imshow('Pixelizer', pixelized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
