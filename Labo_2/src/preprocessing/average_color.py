import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import random
#pip install matplotlib
plt.style.use('_mpl-gallery')
fig, ax = plt.subplots()
moyenne = 0
cpt = 0

def average_color(image):
    average = image.mean(axis=0).mean(axis=0)
    return average

#paths= [
    #'Labo_2/output/clean/Cercles/Cercle2/1_Cercle2.jpg',
    #'Labo_2/output/clean/Hexagones/Hexagone2/289_Hexagone2.jpg',
   # 'Labo_2/output/clean/Hexagones/Hexagone2/127_Hexagone2.jpg'
#]

#images = map(cv2.imread, paths)

#for image in images:
for i in os.listdir('Labo_2/output/clean/Diamants/Diamant5') :
    print(str(i))
    image = cv2.imread('Labo_2/output/clean/Diamants/Diamant5/'+str(i))
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #cv2.imshow('image', grey)

    cv2.waitKey(0)

    average = average_color(grey)

    print(average)

    pixels = np.float32(image.reshape(-1, 3))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
    flags = cv2.KMEANS_RANDOM_CENTERS
    avg_patch = np.ones(shape=image.shape, dtype=np.uint8)*np.uint8(average)
    #cv2.imshow('avg_patch', avg_patch)
    cv2.waitKey(0)
    ax.scatter(random.uniform(1,9), average)
    moyenne=moyenne+average
    cpt = cpt + 1

ax.set(xlim=(0, 10), ylim=(0, 255))
print("la moyenne : " + str(moyenne/cpt))
plt.show()

#cercle2 106,65
#cercle5 115,61


#moyenne pigment gris, nb contours, 