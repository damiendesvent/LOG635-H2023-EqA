import cv2
import json

filesToReview = {
    "C:/Users/aq80980/Desktop/prog/LOG635-H2023-EqA-1/Labo_2/output/Cercles/Cercle2/101_Cercle2.jpg": None
}


if __name__ == '__main__' :
 
    for fileName in filesToReview.keys():
        # Read image
        imageIn = cv2.imread(fileName)
    
        # Select ROI
        rect = cv2.selectROI(img=imageIn, windowName=fileName.split('/')[-1])
        cv2.destroyAllWindows()

        # Save rect
        filesToReview[fileName] = rect

        print(rect)

        # if rect == (0, 0, 0, 0)

        # Crop image
        imgCropped = imageIn[int(rect[1]):int(rect[1]+rect[3]), int(rect[0]):int(rect[0]+rect[2])]

        imgResized = cv2.resize(imgCropped, dsize=(40,40), interpolation=cv2.INTER_AREA)

        outFileRelativePath = '/'.join(fileName.split('/')[-3:])

        print(outFileRelativePath)

        cv2.imwrite('./Labo_2/output/clean/' + outFileRelativePath, imgResized)
    json.dump(filesToReview, open("C:/Users/aq80980/Desktop/prog/LOG635-H2023-EqA-1/Labo_2/output/dict.json",'w'))
