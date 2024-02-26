# пикселизирует фотку 
# TODO - проблемы - все фотки надо предварительно приводить к одному размеру. 

import cv2
import numpy as np

pic_folder ="C:\\Users\\5010858\\Downloads\\Telegram Desktop\\" 

path_l = [  "ola2.jpg",
            "ola.jpg",
            "ola2-removebg-preview.png",
            "photo_2021-10-08_16-10-06.jpg",
            "2_5226645942744125784.jpg", # с гирляндой
            "ataranov.jpg"                              # 5
        ]
path = pic_folder + path_l[5]
scale = 7           # чем меньше тем больше пикселов и качесвеннее итоговая картинка 
color_steps = 5     # шаг цвета чем больше тем качественнее
pic_size = (2,40) # не рабоатает эта херня. 

pix_by_steps = [0 for i in range(color_steps)]

alpha = 0.3
beta = 80

pic, pic2, pic2_ = None, None, None

def updateAlpha(x): 
    global alpha, pic, pic2_
    alpha = cv2.getTrackbarPos('Alpha', 'image')
    alpha = alpha * 0.01
    pic = np.uint8(np.clip((alpha * pic2_ + beta), 0, 255))

def updateBeta(x):
    global beta, pic, pic2_
    beta = cv2.getTrackbarPos('Beta', 'image')
    pic = np.uint8(np.clip((alpha * pic2_ + beta), 0, 255))

def pixelize(img, steps_ = 8):
    step = 255//steps_
    # диапазон 0-255 бьет на квантованные уровни. возвращает уровень
    for i in range(len(img)):
        for j in range(len(img[0])):
            # print(f"line {line}")
            # print(f" img[row][line] {img[row][line]}")
            # print(f" img[row][line] {img[row][line]}")
            for n in range(steps_):
                if n*step < img[i][j] < (n+1)*step:
                    pix_by_steps[n]+=1
                    img[i][j] = n*step
    return img

if __name__ == "__main__":
    pic = cv2.imread(path, 0)
    #cv2.resize(pic, pic_size, pic) # здесь это не работает - см цикл while ниже
    pic2 = cv2.imread(path, 0)
    #cv2.resize(pic2,  pic_size, pic2)
    pic2_ = cv2.imread(path, 0)
    #cv2.resize(pic2_, pic_size, pic2_)
    
    cv2.imshow("image", pic)
    
    # Создать окно и трекбары
    cv2.namedWindow('Esc - Exit')
    cv2.createTrackbar('Alpha', 'image', 0, 300, updateAlpha)
    cv2.createTrackbar('Beta', 'image', 0, 255, updateBeta)
    cv2.setTrackbarPos('Alpha', 'image', 100)
    cv2.setTrackbarPos('Beta', 'image', 50)
    
    #print(f"dims= {dims} shape = {pic.shape}")
    #print(f"dims= {dims} tot number = {tot_pix}")

    #cv2.imshow(f"Scale {scale} quant color {color_steps} size {dims}   tot pix = {tot_pix}  pix set{pix_by_steps}", pic2)
    while (True):

        pic = cv2.imread(path, 0)
        #pic2 = cv2.imread(path, 0)
        #pic2_ = cv2.imread(path, 0)
        cv2.imshow("Esc - Exit", pic)

        dims = pic.shape[:2]
        # pic = cv2.resize(pic, (dims[1]//scale, dims[0]//scale),interpolation = cv2.INTER_BITS)
        # pic = cv2.convertScaleAbs(pic, alpha = 1, beta = 0)

        pic = cv2.resize(pic, (dims[1]//scale, dims[0]//scale),interpolation = cv2.INTER_LINEAR)
        dims = pic.shape[:2]
        tot_pix = dims[0]*dims[1]

        pic = pixelize(pic, color_steps)
        
        pic2 = cv2.resize(pic, (dims[1]*scale, dims[0]*scale), interpolation = cv2.INTER_NEAREST)
        pic2_ = pic2.copy()
        cv2.imshow('image', pic2)
        
        if cv2.waitKey(5) == 27:
            cv2.destroyAllWindows()
            break
cv2.destroyAllWindows()