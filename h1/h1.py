import cv2
import numpy as np
from math import log10, sqrt 

def init() :
    img1 = cv2.imread('img1.png')
    img2 = cv2.resize(img1,(256,256),fx=8,fy=8,interpolation = cv2.INTER_CUBIC)
    cv2.imwrite('img2.png',img2)

def PSNR(img1, img2): 
    mse = np.mean((np.array(img1, dtype=np.float32) - np.array(img2[:32,:32], dtype=np.float32)) ** 2)
    if(mse == 0):  
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse)) 
    return psnr 

def main() :
    init()
    print(PSNR(cv2.imread('img1.png'), cv2.imread('img2.png')))

if __name__ == "__main__":
    main()
