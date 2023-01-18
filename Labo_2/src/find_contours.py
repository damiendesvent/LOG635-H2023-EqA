import cv2
import os

for i in os.listdir('./Labo_2/data') :
    for j in os.listdir('./Labo_2/data/' + str(i)) :
        for k in os.listdir('./Labo_2/data/' + str(i) + '/' + str(j)) :
            if k < 10 :
                image = cv2.imread('./Labo_2/data/' + str(i) + '/' + str(j) + '/' + str(k))
                img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                # apply binary thresholding
                ret, thresh = cv2.threshold(img_gray, 150, 255, cv2.THRESH_BINARY)
                # visualize the binary image
                cv2.imshow('Binary image', thresh)
                cv2.waitKey(0)
                os.makedirs('./Labo_2/output/' + str(i) + '/' + str(j) + '/')
                cv2.imwrite('./Labo_2/output/' + str(i) + '/' + str(j) + '/' + str(k), thresh)
                cv2.destroyAllWindows()