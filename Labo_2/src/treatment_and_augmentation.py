import cv2
import os
from generate_variations import generate_variations

current_dir = './Labo_2/'
data_dir = current_dir + 'output/clean/'
treated_data_dir = current_dir + 'treated/'

for i in os.listdir(data_dir) :
    for j in os.listdir(data_dir + str(i)) :
        for k in os.listdir(data_dir + str(i) + '/' + str(j)) :
            if not os.path.isfile(treated_data_dir + str(i) + '/' + str(j) + '/' + str(k)) :
                print(k)
                image = cv2.imread(data_dir + str(i) + '/' + str(j) + '/' + str(k))
                img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                img_blur = cv2.GaussianBlur(img_gray, (3,3), 0)
                
                os.makedirs(treated_data_dir + str(i) + '/' + str(j) + '/', exist_ok=True)

                for (sufix,img_output) in generate_variations(img_blur) :
                    cv2.imwrite(treated_data_dir + str(i) + '/' + str(j) + '/' + str(k).split('.')[0] + sufix + '.jpg', img_output)
                