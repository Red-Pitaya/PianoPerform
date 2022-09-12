import cv2
import mediapipe as mp
from timeit import default_timer as timer
import numpy as np
import PianoKeys


# 用于绘制
mp_drawing = mp.solutions.drawing_utils

mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# 存放坐标信息




img_length=1000
img_height=600
# For webcam input:
cap = cv2.VideoCapture(1)  #@前为账号密码，@后为ip地址
cap1 = cv2.VideoCapture(2)
cap1.set(cv2.CAP_PROP_FRAME_WIDTH, img_length)
cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, img_height)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, img_length)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, img_height)

round_counter=99
rounds_to_relocate=99
# wkeys_matrix=np.matrix([])
# wkeys_matrix1=np.matrix([])

flag=False

with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
  while(1):
    lmList = []
    lmList1 = []
    tic = timer()
    # 读图像 ##################################################################################################
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    f, img = cap1.read()  # 读取一帧图片
    if not f:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    #进行棋盘检测##############################################################################################
    #is_draw_keypoints = True
    if round_counter == rounds_to_relocate:
        wkey, rectangle= PianoKeys.LocateWhiteKeys(image)
        wkey1, rectangle1= PianoKeys.LocateWhiteKeys(img)
        if wkey==False or wkey1==False:
            print("Keyboard Detection Error!")
            two_view = np.hstack([image, img])
            cv2.imshow("two view", two_view)
            cv2.waitKey(5)
            continue #如果检测出错就会一直卡住不进行下面的动作
        else:
            round_counter=0
            # wkeys_matrix=np.matrix(wkey)
            # wkeys_matrix1=np.matrix(wkey1)
    round_counter=round_counter+1

    #  手指检测 ############################################################################################
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
        h, w, c = image.shape  # 分别存放图像长\宽\通道数
        for handlms in results.multi_hand_landmarks:
            for index, lm in enumerate(handlms.landmark):
                # 索引为0代表手底部中间部位，为4代表手指关键或指尖
                # print(index, lm)  # 输出21个手部关键点的xyz坐标(0-1之间)，是相对于图像的长宽比例
                # 只需使用x和y查找位置信息
                # 将xy的比例坐标转换成像素坐标
                # 中心坐标(小数)，必须转换成整数(像素坐标)
                if index == 8 or index == 12 or index == 16 or index == 20 or index == 4:
                    cx, cy = int(lm.x * w), int(lm.y * h)  # 比例坐标x乘以宽度得像素坐标
                    lmList.append([index, cx, cy])
            mp_drawing.draw_landmarks(
                image,
                handlms,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

# cap1:
    img.flags.writeable = False
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results1 = hands.process(img)

    # Draw the hand annotations on the image.
    img.flags.writeable = True
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if results1.multi_hand_landmarks:
        h, w, c = img.shape
        for handlms in results1.multi_hand_landmarks:
            for index, lm in enumerate(handlms.landmark):
                if index == 8 or index == 12 or index == 16 or index == 20 or index == 4:
                    cx, cy = int(lm.x * w), int(lm.y * h)  # 比例坐标x乘以宽度得像素坐标
                    lmList1.append([index, cx, cy])
                mp_drawing.draw_landmarks(
                    img,
                    handlms,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())
    #  手指距离判断  ################
    fin_key_dis=20**2
    closeList=[]
    closeList1=[]
    if len(lmList)==len(lmList1)==5: #手指都在
        for i in lmList:
            for j in range(len(wkey)):
                dis_square=(i[1]-wkey[j][0])**2+(i[2]-wkey[j][1])**2
                if dis_square<fin_key_dis:
                    closeList.append([i[0],j])
        for i in lmList1:
            for j in range(len(wkey1)):
                dis_square=(i[1]-wkey1[j][0])**2+(i[2]-wkey1[j][1])**2
                if dis_square<fin_key_dis:
                    closeList1.append([i[0],j])
    elif len(lmList)>0 and len(lmList1)>0:
        for i in lmList:
            for j in lmList1:
                if i[0]==j[0]: #同一根手指
                    for k in range(len(wkey)):
                        dis_square = (i[1] - wkey[k][0]) ** 2 + (i[2] - wkey[k][1]) ** 2
                        if dis_square < fin_key_dis:
                            closeList.append([i[0], k])
                    for k in range(len(wkey1)):
                        dis_square = (j[1] - wkey1[k][0]) ** 2 + (j[2] - wkey1[k][1]) ** 2
                        if dis_square < fin_key_dis:
                            closeList1.append([j[0], k])



    ##奏响音乐##########
    if len(closeList)>0 and len(closeList1)>0:
        final_key = [x for x in closeList if x in closeList1]
        for i in final_key:
            print("finger index: %d" %i[0] + " key num:%d" % i[1])
            ###声音：
        print("---------------------------------")
        播放功能写在这里



    # Flip the image horizontally for a selfie-view display.
    #cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
    is_draw_keypoints=True
    if is_draw_keypoints:
        for i in range(len(rectangle)):
            rect_i = rectangle[i]
            image = cv2.circle(image, np.int64(rect_i[0]), 8, (0, 0, 255), -1)
        # 画白色按键重心
        for i in wkey:
            image = cv2.circle(image, np.int64(i), 8, (255, 0, 0), -1)
        for i in range(len(rectangle1)):
            rect_i = rectangle1[i]
            img = cv2.circle(img, np.int64(rect_i[0]), 8, (0, 0, 255), -1)
        # 画白色按键重心
        for i in wkey1:
            img = cv2.circle(img, np.int64(i), 8, (255, 0, 0), -1)
    two_view = np.hstack([image, img])
    cv2.imshow("two view", two_view)


    toc = timer()
    #print(toc - tic)  # 输出的时间，秒为单位
    if cv2.waitKey(5) & 0xFF == 27:
      break

cap1.release()
cap.release()