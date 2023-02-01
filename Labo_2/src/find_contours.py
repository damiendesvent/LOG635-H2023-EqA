import cv2
import os

good_prediction = 0
bad_prediction = 0

for i in os.listdir('./Labo_2/data') :
    for j in os.listdir('./Labo_2/data/' + str(i)) :
        index = 0
        for k in os.listdir('./Labo_2/data/' + str(i) + '/' + str(j)) :
            if not (os.path.isfile('./Labo_2/output/clean/' + str(i) + '/' + str(j) + '/' + str(k)) or os.path.isfile('./Labo_2/output/error/' + str(i) + '/' + str(j) + '/' + str(k))) :
                index += 1
                if index < 10000 :
                    image = cv2.imread('./Labo_2/data/' + str(i) + '/' + str(j) + '/' + str(k))
                    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    img_blur = cv2.GaussianBlur(img_gray, (9,9), 0)
                    edged = cv2.Canny(img_blur, 30,120)

                    contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    img_output = img_gray
                    img_view = image.copy()

                    nb_contours = 0
                    x,y,w,h = 0,0,0,0

                    for c in contours :
                        area = cv2.contourArea(c)
                        if area > 400 :
                            nb_contours += 1
                            x,y,w,h = cv2.boundingRect(c)
                            cv2.rectangle(img_view, (x, y), (x + w, y + h), (36, 255, 12), 2) #crée une bbox verte

                    os.makedirs('./Labo_2/output/clean/' + str(i) + '/' + str(j) + '/', exist_ok=True)
                    os.makedirs('./Labo_2/output/error/' + str(i) + '/' + str(j) + '/', exist_ok=True)

                    if (nb_contours == 1) :
                        print(str(k))
                        rect = cv2.selectROI(img=img_view, windowName=str(k))
                        cv2.destroyAllWindows()
                        if rect == (0,0,0,0) :
                            img_output = image[y:y+h,x:x+w]
                            
                        else :
                            img_output = image[int(rect[1]):int(rect[1]+rect[3]), int(rect[0]):int(rect[0]+rect[2])]
                        img_output = cv2.resize(img_output,dsize=(40,40), interpolation=cv2.INTER_AREA)
                        cv2.imwrite('./Labo_2/output/clean/' + str(i) + '/' + str(j) + '/' + str(k), img_output)
                        good_prediction += 1
                    else :
                        bad_prediction += 1
                        cv2.imwrite('./Labo_2/output/error/' + str(i) + '/' + str(j) + '/' + str(k), img_output)

print("ratio : " + str(good_prediction*100/(good_prediction+bad_prediction))[:5] + " %\n")
print("good : " + str(good_prediction) + "\n")
print("bad : " + str(bad_prediction) + "\n")