import cv2
import numpy as np
import pandas as pd
import function
import utlis

def main():
    webCamFeed = True
    cap = cv2.VideoCapture(1)
    cap.set(10,160)
    og_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    og_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Input resolution: {og_w} x {og_h}")
    heightImg = 480
    widthImg  = 640
    h_rate=og_h/heightImg
    w_rate=og_w/widthImg
    utlis.initializeTrackbars()
    count=0
    while True:

        if webCamFeed:success, img = cap.read()
        ogImage=img.copy()
        img = cv2.resize(img, (widthImg, heightImg))
        imgBlank = np.zeros((heightImg,widthImg, 3), np.uint8)
        config=utlis.valTrackbars() 
        imgGray, imgBlur, imgThreshold, imgDial,imgErode=function.preprocessing(img, config)
        imgContours = img.copy()
        imgBigContour = img.copy()
        contours, hierarchy = cv2.findContours(imgErode, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(imgContours, contours, -1, (255,183,178), 10)
        imgresult,imgresult_=imgBlank,imgBlank
        biggest, maxArea = utlis.biggestContour(contours)
        if biggest.size != 0:
            biggest=utlis.reorder(biggest)
            cv2.drawContours(imgBigContour, biggest, -1, (255,183,178), 20)
            imgBigContour = utlis.drawRectangle(imgBigContour,biggest,2)
            pts1 = np.float32(biggest)
            pts2 = np.float32([[0, 0],[widthImg, 0], [0, heightImg],[widthImg, heightImg]])
            matrix = cv2.getPerspectiveTransform(pts1, pts2)
            imgWarpColored = cv2.warpPerspective(img, matrix, (widthImg, heightImg))
            imgWarpColored = cv2.resize(imgWarpColored,(widthImg,heightImg))

            Ogbiggest=utlis.Ogcontour(biggest,w_rate,h_rate)

            ogImage=utlis.drawRectangle(ogImage,Ogbiggest,4)
            cv2.drawContours(ogImage, Ogbiggest, -1, (0,0,255), 20)
            imgresult,imgresult_=function.gen_results(ogImage,Ogbiggest)
            imgWarpGray = cv2.cvtColor(imgWarpColored,cv2.COLOR_BGR2GRAY)
            imgAdaptiveThre= cv2.adaptiveThreshold(imgWarpGray, 255, 1, 1, 7, 2)
            imgAdaptiveThre = cv2.bitwise_not(imgAdaptiveThre)
            imgAdaptiveThre=cv2.medianBlur(imgAdaptiveThre,3)

            imageArray = ([img ,imgGray, imgBlur],
                        [imgThreshold,imgErode,imgContours],
                        [imgBigContour,imgWarpColored,imgAdaptiveThre])

        else:

            imageArray = ([img ,imgGray, imgBlur],
                        [imgThreshold,imgErode,imgContours],
                        [imgBlank, imgBlank, imgBlank])

        # lables = [["Blured&Gray","Threshold","Contours"],
        #             ["Biggest Contour","Warp Prespective","Adaptive Threshold"]]

        stackedImage = utlis.stackImages(imageArray,0.75)
        cv2.imshow("Result",stackedImage)
        cv2.imshow("original", ogImage)
        if cv2.waitKey(1) & 0xFF == ord('b'):
            cv2.imwrite("/Users/mac/Dev/Computer_Vision/Computer-Vision/ouput/vertical/myImage"+str(count)+".jpg",imgresult)
            cv2.imwrite("/Users/mac/Dev/Computer_Vision/Computer-Vision/ouput/horiziontal/myImage"+str(count)+".jpg",imgresult_)
            cv2.imshow('Result', stackedImage)
            cv2.waitKey(300)
            count += 1
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()