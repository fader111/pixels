""" отсюда https://nuancesprog.ru/p/12462/?ysclid=lt8mwcw434232935997"""
import cv2
import numpy as np
from misc import *
import yaml

def edge_mask(img, line_size, blur_value):
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  gray_blur = cv2.medianBlur(gray, blur_value)
  edges = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, line_size, blur_value)
  return edges

line_size = 7
blur_value = 7

path_ = r"C:\Users\Anton\Pictures\IMG_20210908_163738.png"


img = cv2.imread(path_)
img = resizeAndPad(img, (600,600))
edges = edge_mask(img, line_size, blur_value)
cv2.imshow('edges', edges)

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

total_color = 9
img = color_quantization(img, total_color)
cv2.imshow('cquant', img)

blurred = cv2.bilateralFilter(img, d=7, sigmaColor=200,sigmaSpace=200)

cartoon = cv2.bitwise_and(blurred, blurred, mask=edges)

cv2.imshow('blured', blurred)

cartoon = cv2.bitwise_and(blurred, blurred, mask=edges)
cv2.imshow('cartoon', cartoon)

cv2.waitKey(0)