import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('lena.png')

def main() :
    x = input('enter a method : ')
    print(x)
    if x == 'historgram' :
        historgram()
    elif x == 'laplace' :
        Laplace()
    elif x == 'gamma' :
        gamma()

#直方图
def historgram() :
    img = cv2.imread('lena.png', cv2.CV_16UC1)

    imgEqualizeHist = cv2.equalizeHist(img)
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    imgCLAHE = clahe.apply(img)
    
    cv2.imshow('origin', img)
    #均衡化
    cv2.imshow('equalizeHist', imgEqualizeHist)
    #CLAHE
    cv2.imshow('createCLAHE', imgCLAHE)
    
    cv2.imshow("hist", img)
    cv2.waitKey()
    cv2.destroyAllWindows()
    
#拉普拉斯
def Laplace() :
    gray_lap = cv2.Laplacian(img,cv2.CV_16S,ksize = 3)
    dst = cv2.convertScaleAbs(gray_lap)
    
    cv2.imshow('laplacian',dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#伽码
def gamma() :
    gamma = 2
    img2 = np.power(img/float(np.max(img)), gamma)

    cv2.imshow('gamma',img2)
    cv2.waitKey(0)




if __name__ == "__main__":
    main()   
