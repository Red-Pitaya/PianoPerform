import cv2
import numpy as np
import math

res_box=[]

def IoU_select(box,img_size,iou_thrd=0.7): #用于筛选，但是不怎么用
    global res_box
    if len(res_box)==0:
        res_box.append(box)
        return True

    color = (1, 0, 0)
    color2 = (0, 1, 0)
    area_box = cv2.contourArea(box)
    for b in res_box:
        area_b=cv2.contourArea(b)
        img = np.zeros([img_size[0], img_size[1], 3])
        img = cv2.drawContours(img, [box], 0, color, thickness=-1)
        img = cv2.drawContours(img, [b], 0, color2, thickness=-1)
        union_area = img.sum()
        inter_area = area_box + area_b - union_area
        IOU = inter_area / union_area
        if IOU>iou_thrd:
            return False
    res_box.append(box)
    return True

#选取正确的轮廓（黑色按键的）
def Contours_Select(img,contours,min_area=500,max_area=2000,ratio=0.8,figure=False,distance=10,file_name="",num_keys=10):
    res_cnt=[] #清洗后剩余的轮廓
    center_xy=np.matrix([0,0]) #经过检验的外接矩形中心点（筛除重合度过高的contours）
    for cnt in contours:
        #面积筛选
        area = cv2.contourArea(cnt)
        if area<min_area or area>max_area:
            continue
        else:
            #外接矩形筛选，四个指标：（1）contour要逼近外接矩形的面积 (2)长宽比正确
            #（2）筛除过于靠近的contour （4）不能和已有的contour重合度过高
            rect = cv2.minAreaRect(cnt)
            xy_temp=np.matrix(rect[0])  #外接矩形的中心点
            box = cv2.boxPoints(rect)
            area_box=cv2.contourArea(box) #外接矩形的面积

            if area/area_box<ratio or max(rect[1])/min(rect[1])<3 or min(np.linalg.norm(center_xy-xy_temp,axis=1))<distance:
                continue
            #绘图（如果需要的话）
            box = np.int64(box)
            if figure:
                img= cv2.drawContours(img, [box], 0, (0, 0, 255), 3)
            #输出
            #print("area:%d poly area:%d" % (area, area_box))
            #检验通过，更新结果
            center_xy=np.append(center_xy,xy_temp,axis=0)
            res_cnt.append(rect)
    if figure:
        cv2.namedWindow("BlackKeys", 0)
        cv2.resizeWindow("BlackKeys", 640, 480)
        cv2.imshow('BlackKeys', img)
        #cv2.imwrite(file_name+"boxes.jpg",img)
    if len(res_cnt)!=num_keys:
        return False
    res_cnt=sorted(res_cnt, key=lambda x: (x[0][0]))
    return res_cnt


def Get_whiteKey(rectangle,direction=1,move_down=0.95,num_keys=10): #获取白色按键的位置
    if len(rectangle)!=10:
        return False
    wkey=[]
    #首先获取黑键下方的点
    for rect_i in rectangle:
        if rect_i[1][0]>rect_i[1][1]: #第一个触碰到的是长边，按键向左倾倒
            cosine = math.cos(math.radians(rect_i[2]))
            sine = math.sin(math.radians(rect_i[2]))
            wkey_i=(rect_i[0][0]-direction*cosine*rect_i[1][0],rect_i[0][1]-sine*direction*move_down*rect_i[1][0])
            wkey.append(wkey_i)
        else:
            cosine = math.cos(math.radians(90-rect_i[2]))
            sine = math.sin(math.radians(90-rect_i[2]))
            wkey_i = (rect_i[0][0] + direction*cosine * rect_i[1][1], rect_i[0][1] - sine *direction*move_down* rect_i[1][1])
            wkey.append(wkey_i)
    result_keys=[]
    wkey2=[]
    wkey2.append(wkey[0])
    for i in range(0,len(rectangle)-1):
        wkey2.append(wkey[i])
    wkey=np.matrix(wkey) #坐标
    wkey2=np.matrix(wkey2)
    vectors=0.5*(wkey-wkey2)  #向量，从行1开始有效
    if direction == 1 and len(rectangle)==num_keys: #位置逻辑关系：向量相加：
        result_keys.append(tuple(np.array(wkey[0]-vectors[1])[0]))
        result_keys.append(tuple(np.array(wkey[0] + vectors[1])[0]))
        result_keys.append(tuple(np.array(wkey[1] + vectors[2])[0]))
        result_keys.append(tuple(np.array(wkey[2] + 0.5*vectors[3])[0]))
        result_keys.append(tuple(np.array(wkey[3] - vectors[4])[0]))
        result_keys.append(tuple(np.array(wkey[3] + vectors[4])[0]))
        result_keys.append(tuple(np.array(wkey[4] + 0.5*vectors[5])[0]))
        result_keys.append(tuple(np.array(wkey[3] + vectors[4])[0]))
        result_keys.append(tuple(np.array(wkey[5] - vectors[6])[0]))
        result_keys.append(tuple(np.array(wkey[5] + vectors[6])[0]))
        result_keys.append(tuple(np.array(wkey[6] + vectors[7])[0]))
        result_keys.append(tuple(np.array(wkey[7] + 0.5 * vectors[8])[0]))
        result_keys.append(tuple(np.array(wkey[8] + vectors[9])[0]))
        result_keys.append(tuple(np.array(wkey[8] - vectors[9])[0]))
        result_keys.append(tuple(np.array(wkey[9] + vectors[9])[0]))
    elif direction == -1 and len(rectangle)==num_keys:
        result_keys.append(tuple(np.array(wkey[0] - vectors[1])[0]))
        result_keys.append(tuple(np.array(wkey[0] + vectors[1])[0]))
        result_keys.append(tuple(np.array(wkey[1] + 0.5*vectors[2])[0]))
        result_keys.append(tuple(np.array(wkey[2] - vectors[3])[0]))
        result_keys.append(tuple(np.array(wkey[2] + vectors[3])[0]))
        result_keys.append(tuple(np.array(wkey[3] + vectors[4])[0]))
        result_keys.append(tuple(np.array(wkey[4] + 0.5*vectors[5])[0]))
        result_keys.append(tuple(np.array(wkey[5] - vectors[6])[0]))
        result_keys.append(tuple(np.array(wkey[5] + vectors[6])[0]))
        result_keys.append(tuple(np.array(wkey[6] + 0.5*vectors[7])[0]))
        result_keys.append(tuple(np.array(wkey[7] - vectors[8])[0]))
        result_keys.append(tuple(np.array(wkey[7] + vectors[8])[0]))
        result_keys.append(tuple(np.array(wkey[8] + vectors[9])[0]))
        result_keys.append(tuple(np.array(wkey[9] + vectors[9])[0]))
    else:
        return False
    return result_keys


# # Read the original image
# filename="5"
# img = cv2.imread(filename+".jpg")
def LocateWhiteKeys(img):
    # Convert to graycsale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Blur the image for better edge detection
    img_blur = cv2.blur(img_gray, (3,3))

    # Sobel Edge Detection
    sobelx = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5) # Sobel Edge Detection on the X axis
    sobely = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5) # Sobel Edge Detection on the Y axis
    sobelxy = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5) # Combined X and Y Sobel Edge Detection
    edges = cv2.Canny(image=img_blur, threshold1=50, threshold2=150) # Canny Edge Detection
    #threshold1、2是重要参数！！

    #查看边缘检测结果
    # cv2.imshow('rect', edges)
    # cv2.waitKey(0)
    #进行轮廓检测
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    rectangle=Contours_Select(img,contours,figure=False)
    if rectangle==False:
        return [False,False]
    wkey = Get_whiteKey(rectangle,direction=1,move_down=0.8)
    if wkey==False:
        return [False,False]
    else:
        return [wkey,rectangle]


#一些也许可以使用的计算代码
#1.计算重心
# M=cv2.moments(box)
# cx = int(M['m10'] / M['m00'])  # 重心的x坐标
# cy = int(M['m01'] / M['m00'])
#            print("%d %d"%(cx,cy))

# 第二道检查：拟合多边形
# epsilon = ratio * cv2.arcLength(cnt, True)
# approx = cv2.approxPolyDP(cnt, epsilon, True)
# img=cv2.polylines(img, [approx], True, (0, 0, 255), 2)
# area_approx = cv2.contourArea(approx)
# if len(approx)<4 or area_approx/area<0.9:
#     continue
# print("area:%d poly area:%d"%(area,area_approx))

# cv2.namedWindow("rect", 0)
# cv2.resizeWindow("rect", 640, 480)
# cv2.imshow('rect', img)
# cv2.waitKey(0)