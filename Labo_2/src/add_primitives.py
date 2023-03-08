import cv2

def number_items(image, mode='canny') :
    if mode == 'canny' :
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _,image_ts = cv2.threshold(image_gray, 30, 255, cv2.THRESH_BINARY_INV)
        image_blur = cv2.GaussianBlur(image_gray, (3,3), 0)
        image_canny = cv2.Canny(image_blur,20,150,3)
        image_dilate = cv2.dilate(image_canny, (1,1), iterations=0)
        contours,_ = cv2.findContours(image_dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_clean = []
        for contour in contours :
            area = cv2.contourArea(contour)
            print(area)
            if area < 1400 and area > 1:
                contours_clean.append(contour)
                x,y,w,h = cv2.boundingRect(contour)
                cv2.rectangle(image_dilate, (x, y), (x + w, y + h), (36, 255, 12), 2)
        image_out = cv2.drawContours(image,contours_clean, -1, (255,0,0))
        return image_out,len(contours_clean)

    elif mode == 'threeshold' :
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _,image_ts = cv2.threshold(image_gray, 30, 255, cv2.THRESH_BINARY_INV)
        contours,_ = cv2.findContours(image_ts, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_clean = []
        for contour in contours :
            area = cv2.contourArea(contour)
            if area < 400 and area > 10:
                contours_clean.append(contour)
        return image_ts,len(contours_clean)

    else :
        raise NameError('Le mode n\'est pas reconnu')



img = cv2.imread('Labo_2/output/clean/Cercles/Cercle5/12_Cercle5.jpg')

#print(number_items(img)[1])

img_ts = number_items(img)[0]
cv2.imshow('hello', img_ts)
cv2.waitKey(0)