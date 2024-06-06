import cv2
import numpy as np
import utlis


def preprocessing(image, config):
    blur_layers, threshold_1, threshold_2 = config
    imgGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    imgBlur=imgGray
    for layer in range(blur_layers):
        imgBlur = cv2.GaussianBlur(imgBlur, (5, 5), 1)
    imgThreshold = cv2.Canny(imgBlur,threshold_1, threshold_2)
    # _,imgThreshold=cv2.threshold(imgBlur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # imgThreshold = cv2.bitwise_not(imgThreshold)
    kernel = np.ones((5, 5))
    imgDial = cv2.dilate(imgThreshold, kernel, iterations=2)
    imgErode = cv2.erode(imgDial, kernel, iterations=1)
    return imgGray, imgBlur, imgThreshold, imgDial,imgErode




def gen_results(ogImage,Ogbiggest):
    resultH=297*8
    resultW=210*8
    pts1_=np.float32(Ogbiggest)
    pts2_ = np.float32([[0, 0],[resultW, 0], [0, resultH],[resultW, resultH]])
    matrix_ = cv2.getPerspectiveTransform(pts1_, pts2_)
    imgresult = cv2.warpPerspective(ogImage, matrix_, (resultW,resultH))
    imgresult= cv2.resize(imgresult,(resultW,resultH))
    imgresult = cv2.cvtColor(imgresult,cv2.COLOR_BGR2GRAY)
    imgresult=imgresult[20:imgresult.shape[0] - 20, 20:imgresult.shape[1] - 20]
    imgresult= cv2.adaptiveThreshold(imgresult, 255, 1, 1, 15, 2)
    # _,imgresult=cv2.threshold(imgresult, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    imgresult = cv2.bitwise_not(imgresult)
    kernel = np.ones((5, 5))
    imgresult = cv2.dilate(imgresult, kernel, iterations=3)
    imgresult = cv2.erode(imgresult, kernel, iterations=1)
    imgresult=cv2.medianBlur(imgresult,3)

    resultW=297*8
    resultH=210*8
    pts1_=np.float32(Ogbiggest)
    pts2_ = np.float32([[0, 0],[resultW, 0], [0, resultH],[resultW, resultH]])
    matrix_ = cv2.getPerspectiveTransform(pts1_, pts2_)
    imgresult_ = cv2.warpPerspective(ogImage, matrix_, (resultW,resultH))
    imgresult_= cv2.resize(imgresult_,(resultW,resultH))
    imgresult_ = cv2.cvtColor(imgresult_,cv2.COLOR_BGR2GRAY)
    imgresult_=imgresult_[20:imgresult_.shape[0] - 20, 20:imgresult_.shape[1] - 20]
    imgresult_= cv2.adaptiveThreshold(imgresult_, 255, 1, 1, 15, 2)
    # _,imgresult_=cv2.threshold(imgresult_, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    imgresult_ = cv2.bitwise_not(imgresult_)
    kernel = np.ones((5, 5))
    imgresult = cv2.dilate(imgresult_, kernel, iterations=3)
    imgresult = cv2.erode(imgresult_, kernel, iterations=1)
    imgresult_=cv2.medianBlur(imgresult_,3)

    return imgresult, imgresult_

