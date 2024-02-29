import cv2
import numpy as np
from misc import *


# Callback function for trackbar
def update_brightness(value):
    global brightness
    brightness = value
    apply_brightness_contrast()

def update_contrast(value):
    global contrast
    contrast = value
    apply_brightness_contrast()

def apply_brightness_contrast():
    # Apply brightness and contrast
    adjusted = np.int16(image) * (contrast/127 + 1) - contrast + brightness
    adjusted = np.clip(adjusted, 0, 255)
    adjusted = np.uint8(adjusted)

    # Display the adjusted image
    cv2.imshow('Adjusted Image', adjusted)

# Read the image from path
path_ = r"C:\Users\Anton\Pictures\IMG_20210908_163738.png"

image = cv2.imread(path_)
image = resizeAndPad(image, (600, 600))
cv2.imshow('Original Image', image)

# Create a window
cv2.namedWindow('Adjusted Image')

# Create trackbars for adjusting brightness and contrast
brightness = 0
contrast = 127
cv2.createTrackbar('Brightness', 'Adjusted Image', brightness, 255, update_brightness)
cv2.createTrackbar('Contrast', 'Adjusted Image', contrast, 255, update_contrast)

# Apply initial brightness and contrast
apply_brightness_contrast()

while True:
    # Exit the loop if 'Esc' is pressed
    if cv2.waitKey(1) == 27:
        break

# Close all windows
cv2.destroyAllWindows()
