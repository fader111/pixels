import cv2
import numpy as np


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
    output = np.repeat(np.repeat(blocks, pixel_size,
                       axis=1), pixel_size, axis=0)
    # Remove padding
    return output[:height, :width].astype("uint8")


def on_trackbar_change(value):
    pixel_size = cv2.getTrackbarPos('Pixel Size', 'Pixelizer')
    pixelized = pixeller(img, pixel_size=pixel_size)
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

if __name__ == '__main__':
    img = cv2.imread(path_)
    pixel_size = 5
    pixelized = pixeller(img, pixel_size=pixel_size)

    cv2.namedWindow('Pixelizer')
    cv2.createTrackbar('Pixel Size', 'Pixelizer',
                       pixel_size, 50, on_trackbar_change)

    cv2.imshow('Pixelizer', pixelized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
