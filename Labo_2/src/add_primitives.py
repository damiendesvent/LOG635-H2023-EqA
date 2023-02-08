import cv2

def primitive_threshold(image) :
    _,image_ts = cv2.threshold(image, 50, 255, cv2.THRESH_BINARY_INV)
    return image_ts

img = cv2.imread('Labo_2/treated/Diamants/Diamant5/1_Diamant5_270.jpg')
img_ts = primitive_threshold(img)
cv2.imshow('hello', img_ts)
cv2.waitKey(0)