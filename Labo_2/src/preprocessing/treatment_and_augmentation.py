import cv2
import os

current_dir = './Labo_2/'
data_dir = current_dir + 'output/clean/'
treated_data_dir = current_dir + 'treated/'

for i in os.listdir(data_dir) :
    for j in os.listdir(data_dir + str(i)) :
        for k in os.listdir('./Labo_2/data/' + str(i) + '/' + str(j)) :
            if not os.path.isfile(treated_data_dir + str(i) + '/' + str(j) + '/' + str(k)) and k == '1_Diamant2.jpg':
                print(k)
                image = cv2.imread(data_dir + str(i) + '/' + str(j) + '/' + str(k))
                img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                img_blur = cv2.GaussianBlur(img_gray, (3,3), 0)
                
                os.makedirs(treated_data_dir + str(i) + '/' + str(j) + '/', exist_ok=True)

                cv2.imwrite(treated_data_dir + str(i) + '/' + str(j) + '/' + str(k), img_blur)
                img_rotate_90 = cv2.rotate(img_blur, cv2.ROTATE_90_CLOCKWISE)
                cv2.imwrite(treated_data_dir + str(i) + '/' + str(j) + '/' + str(k).split('.')[0] + '_90.jpg', img_rotate_90)
                img_rotate_180 = cv2.rotate(img_rotate_90, cv2.ROTATE_90_CLOCKWISE)
                cv2.imwrite(treated_data_dir + str(i) + '/' + str(j) + '/' + str(k).split('.')[0] + '_180.jpg', img_rotate_180)
                img_rotate_270 = cv2.rotate(img_rotate_180, cv2.ROTATE_90_CLOCKWISE)
                cv2.imwrite(treated_data_dir + str(i) + '/' + str(j) + '/' + str(k).split('.')[0] + '_270.jpg', img_rotate_270)

                