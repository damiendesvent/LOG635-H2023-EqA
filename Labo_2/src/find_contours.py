import cv2
import os

for i in os.listdir('./Labo_2/data') :
    for j in os.listdir('./Labo_2/data/' + str(i)) :
        index = 0
        for k in os.listdir('./Labo_2/data/' + str(i) + '/' + str(j)) :
            index += 1
            if index < 5 :
                image = cv2.imread('./Labo_2/data/' + str(i) + '/' + str(j) + '/' + str(k))
                img_reduce = cv2.resize(image, dsize=(320,140), interpolation=cv2.INTER_AREA)
                img_gray = cv2.cvtColor(img_reduce, cv2.COLOR_BGR2GRAY)
                # apply binary thresholding
                #thresh = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 4, 2)
                img_blur = cv2.GaussianBlur(img_gray, (9,9), 0)
                edged = cv2.Canny(img_blur, 30,200)
                #tr, edged = cv2.threshold(img_blur, 100, 255, cv2.THRESH_BINARY_INV)

                contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                img_output = img_reduce
                for c in contours :
                    area = cv2.contourArea(c)
                    if area > 300 :
                        #img_output = cv2.drawContours(img_reduce, contours, -1, (0,255,0), 1)
                        x,y,w,h = cv2.boundingRect(c)
                        cv2.rectangle(img_output, (x, y), (x + w, y + h), (36, 255, 12), 2)

                os.makedirs('./Labo_2/output/' + str(i) + '/' + str(j) + '/', exist_ok=True)
                cv2.imwrite('./Labo_2/output/' + str(i) + '/' + str(j) + '/' + str(k), img_output)
                #print(contours)