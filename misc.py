import cv2
import numpy as np

def resizeAndPad__(img, size, padColor=255):

    h, w = img.shape[:2]
    sh, sw = size

    # interpolation method
    if h > sh or w > sw: # shrinking image
        interp = cv2.INTER_AREA

    else: # stretching image
        interp = cv2.INTER_CUBIC

    # aspect ratio of image
    aspect = float(w)/h 
    saspect = float(sw)/sh

    if (saspect > aspect) or ((saspect == 1) and (aspect <= 1)):  # new horizontal image
        new_h = sh
        new_w = np.round(new_h * aspect).astype(int)
        pad_horz = float(sw - new_w) / 2
        pad_left, pad_right = np.floor(pad_horz).astype(int), np.ceil(pad_horz).astype(int)
        pad_top, pad_bot = 0, 0

    elif (saspect < aspect) or ((saspect == 1) and (aspect >= 1)):  # new vertical image
        new_w = sw
        new_h = np.round(float(new_w) / aspect).astype(int)
        pad_vert = float(sh - new_h) / 2
        pad_top, pad_bot = np.floor(pad_vert).astype(int), np.ceil(pad_vert).astype(int)
        pad_left, pad_right = 0, 0

    # set pad color
    if len(img.shape) is 3 and not isinstance(padColor, (list, tuple, np.ndarray)): # color image but only one color provided
        padColor = [padColor]*3

    # scale and pad
    scaled_img = cv2.resize(img, (new_w, new_h), interpolation=interp)
    scaled_img = cv2.copyMakeBorder(scaled_img, pad_top, pad_bot, pad_left, pad_right, borderType=cv2.BORDER_CONSTANT, value=padColor)

    return scaled_img

def resizeAndPad(img, size, padColor=255):

    h, w = img.shape[:2]
    sh, sw = size

    # interpolation method
    if h > sh or w > sw: # shrinking image
        interp = cv2.INTER_AREA
    else: # stretching image
        interp = cv2.INTER_CUBIC

    # aspect ratio of image
    aspect = w/h  # if on Python 2, you might need to cast as a float: float(w)/h

    # compute scaling and pad sizing
    if aspect > 1: # horizontal image
        new_w = sw
        new_h = np.round(new_w/aspect).astype(int)
        pad_vert = (sh-new_h)/2
        pad_top, pad_bot = np.floor(pad_vert).astype(int), np.ceil(pad_vert).astype(int)
        pad_left, pad_right = 0, 0
    elif aspect < 1: # vertical image
        new_h = sh
        new_w = np.round(new_h*aspect).astype(int)
        pad_horz = (sw-new_w)/2
        pad_left, pad_right = np.floor(pad_horz).astype(int), np.ceil(pad_horz).astype(int)
        pad_top, pad_bot = 0, 0
    else: # square image
        new_h, new_w = sh, sw
        pad_left, pad_right, pad_top, pad_bot = 0, 0, 0, 0

    # set pad color
    if len(img.shape) is 3 and not isinstance(padColor, (list, tuple, np.ndarray)): # color image but only one color provided
        padColor = [padColor]*3

    # scale and pad
    scaled_img = cv2.resize(img, (new_w, new_h), interpolation=interp)
    scaled_img = cv2.copyMakeBorder(scaled_img, pad_top, pad_bot, pad_left, pad_right, borderType=cv2.BORDER_CONSTANT, value=padColor)

    return scaled_img

def color_quantization(img, k):
    # Преобразуем изображение
    data = np.float32(img).reshape((-1, 3))

    # Задаем критерии
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.001)

    # Внедряем метод k-средних
    ret, label, center = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    result = center[label.flatten()]
    result = result.reshape(img.shape)
    return result


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


def pixeller(img, custom_colors, pixel_size=5, min_pixel_size=3):
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


def get_pixels_num(img_size, pixel_size):
    pixels_num_w, pixels_num_h = img_size[0]//pixel_size, img_size[1]//pixel_size
    print (f"w/h pixels {pixels_num_w, pixels_num_h} tot -{pixels_num_w * pixels_num_h} pxl size -{pixel_size}" )
    return pixels_num_w, pixels_num_h
