import cv2
import numpy as np #导入库
img = cv2.imread('1.png')  #需要识别的图片也可以是视频

img = cv2.medianBlur(img, 5)  # 中值滤波，去除黑色边际中可能含有的噪声干扰

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
lower_blue = np.array([0,0,0])
upper_blue = np.array([180,43,220])

erode_hsv = cv2.erode(hsv, None, iterations=2)
inRange_hsv = cv2.inRange(erode_hsv, lower_blue, upper_blue)

# Creating kernel
kernel = np.ones((6, 6), np.uint8)
inRange_hsv = cv2.erode(inRange_hsv, kernel, cv2.BORDER_REFLECT, iterations=3)

# 和原始图片进行and操作，获得黑色区域
res = cv2.bitwise_and(img,img, mask= inRange_hsv)
cv2.imshow("thres", res)

ret,thresh = cv2.threshold(inRange_hsv,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

thresh = ~thresh
# 查看二值化结果
cv2.imshow("thres", thresh)
#cv2.imwrite("thres.jpg", thresh)

contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

img_contours = []
for i in range(len(contours)):
    img_temp = np.zeros(img.shape, np.uint8)
    img_contours.append(img_temp)

    area = cv2.contourArea(contours[i], False)
    if area > 1000:
        print("轮廓%d的面积是: %d" % (i, area))
        cv2.drawContours(img_contours[i], contours, i, (0, 0, 255), 5)
        #cv2.imshow("contours %d" % i, img_contours[i])

cv2.imshow('mask',inRange_hsv)
cv2.imshow('img',img)

cv2.waitKey(0)
