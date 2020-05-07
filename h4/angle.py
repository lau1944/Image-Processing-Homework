import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('./angle.jpg')
Z = img.reshape((-1, 3))
# convert to np.float32
Z = np.float32(Z)


criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 8
ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

for y in range(len(center)):
    a1 = []
    for i,x in enumerate(label.ravel()):
        if x==y:
            a1.append(list(Z[i]))
        else:
            a1.append([255,255,255])
    a2=np.array(a1)
    a3=a2.reshape((img.shape))


cv2.imwrite('./angle2.jpg', a3)


img2 = cv2.imread('./angle2.jpg')
print(cv2.norm(img, img2, cv2.NORM_INF, mask=None))
print(cv2.norm(img, img2, cv2.NORM_L1, mask=None))
print(cv2.norm(img, img2, cv2.NORM_L2, mask=None))