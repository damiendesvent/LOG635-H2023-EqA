import numpy as np
import cv2

def average_color(image):
    average = image.mean(axis=0).mean(axis=0)
    return average

paths= [
    'C:/Users/aq80980/Desktop/prog/LOG635-H2023-EqA/Labo_2/data/Cercles/Cercle2/1_Cercle2.jpg',
    'C:/Users/aq80980/Desktop/prog/LOG635-H2023-EqA/Labo_2/data/Hexagones/Hexagone2/289_Hexagone2.jpg',
    'C:/Users/aq80980/Desktop/prog/LOG635-H2023-EqA/Labo_2/data/Hexagones/Hexagone2/127_Hexagone2.jpg'
]

images = map(cv2.imread, paths)

for image in images:

    cv2.imshow('image', image)

    cv2.waitKey(0)

    average = average_color(image)

    print(average)

    pixels = np.float32(image.reshape(-1, 3))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
    flags = cv2.KMEANS_RANDOM_CENTERS
    avg_patch = np.ones(shape=image.shape, dtype=np.uint8)*np.uint8(average)
    cv2.imshow('avg_patch', avg_patch)
    cv2.waitKey(0)