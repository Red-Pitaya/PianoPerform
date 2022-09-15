### Introduction
Based on OpenCV and media apipe library, it implements a portable smart piano. The camera recognizes the position of the corresponding button of the finger, emits the corresponding tone, and displays the corresponding note on the host computer.

![在这里插入图片描述](https://img-blog.csdnimg.cn/2fc2eb7ce5004fa59d7f6cd38023014f.png)

Positioning and 3D reconstruction using binocular cameras; Determine whether to press it by judging the position of the hand relative to the paper keys, and the sound is emitted by the sound equipment brought by the computer, and the corresponding scale is indicated.

### Background
对于一些落后地区，由于资金不足，音乐课的教学成为了很大的一个问题。再而对于钢琴这种极其昂贵的乐器，购买更是一个不可能实现的事情。
但随着现代自动化程度的不断提高，计算机技术的普及，对于钢琴的使用也可以朝着自动化、智能化的方向发展。通过计算机的控制，实现纸质钢琴的音调播放及教学无疑就成为一项重要课题；开发便捷，对落后地区的音乐教学有重要的现实意义。  

### Development
 - python-opencv
 - midiapipe
 - pygame

### 实现过程
####  1、手势识别
1.1 mediapipe 原理   
 Mediapipe Hands是一种高保真的手和手指跟踪解决方案，利用机器学习从一帧中推断出21个手关节的3D坐标。
 ![在这里插入图片描述](https://img-blog.csdnimg.cn/f79f7abc18d442718a361a0afa275ba2.png)
 1. Mediapipe 处理的是RGB的img，但是摄像头侦测的图片都是BGR，所以需要进行转换。
 2. 将转换的图片放到hand处理程序中进行处理，将侦测到的21个关节点的手部坐标(x,y)打印出来。
 3. 利用mediapipe提供的draw功能将坐标点画出来。

1.2 检测手指坐标
通过VideoCaputure（）调用电脑的摄像头，将mediapipe库手部识别结果遍历，得到每个手指的坐标，然后结果存到一个数组中。Mediapipe库的检测结果是归一化的转换为真实的像素坐标。

####  2、琴键识别
**2.1 canny边缘检测**

   **2.1.1 前期处理**

 - 读入图像

 第一步使用OpenCV中的imread() 函数将图像读入内存。
这里，我们按照灰度图格式将彩色图片读入内存，这是因为在检测边缘是你不需要彩色信息。

 - 图像平滑

 载入图像之后，使用 Blur() 函数对其进行平滑。这是为了去除图片中的噪声。在边缘检测中，需要对像素亮度进行数值求导，结果会产生带有噪声的边缘。换而言之，图像中的相邻像素的亮度（特别是在边缘附近）波动比较大，者会被我们当成边界，但并不是我们寻找的主要边缘结构。
对于边界附近的亮度通过模糊化进行平滑，可以比较容易识别出图像中的主要边缘结构。 采用均值滤波：它只取内核区域下所有像素的平均值并替换中心元素。
  3x3标准化的盒式过滤器如下所示：
![在这里插入图片描述](https://img-blog.csdnimg.cn/1a4de22a5dff451fb2349e2c8b562fb0.png)

**2.1.2 进行边缘检测**

    Canny边缘检测是当今最流行的边缘检测算法，这是因为它具有可靠性和灵活性。算法通过三个步骤抽取原始图像的边缘信息。 除了使用图像模糊化，还有一些其他必要的去除噪声的预处理过程。加上这一步骤，Canny算法就具有四个处理阶段了：
 - 消除噪声；
 - 计算图像的亮度梯度值；
 - 减除虚假边缘； 
 - 带有滞回的阈值检测；

1、去除噪声
   原始图像的像素通常会产生边缘噪声，所以Canny边缘检测在计算边缘之前进行噪声去除非常重要。使用高斯模糊化可以去除大部分，或者尽量减少不必要的图像细节产生不必要的边缘.
2、计算图像梯度
  在图像平滑后，通过使用Sobel水平和垂直卷积核对图像进行滤波。利用滤波结果可以同时计算出亮度梯度的幅值（ G）和方向（θ）， 下面给出了计算方法：
       ![在这里插入图片描述](https://img-blog.csdnimg.cn/a3e85415d3974e94875497f275f410b0.png)

  梯度方向被量化的最接近的45°倍数的数值。
  3、去除虚假边缘
  完成了图像中的噪声去除和亮度梯度计算，算法的这个步骤通过使用 non-maximum supperssion 来剔除不需要的像素（这些像素不是组成边缘的部分）。 通过比较每个像素与周围像素在水平和垂直两个方向上的梯度值来实现剔除虚假边缘。如果一个像素对应的梯度在局部是最大的，也就是比他的上下左右像素梯度都大，它就保留下来。否则将就该像素置为0. 下面图像显示了处理结果。
4、带有滞回的阈值处理
  最后一步，梯度值与两个阈值进行比较。其中一个小于另外一个。  
 - 如果图像梯度值比较大的阈值还大，代表这个像素是一个很强的边缘，它被保留在最后的边缘图中；
 - 如果梯度幅值小于较小的阈值，则该像素被抑制，从最终边缘图中去除；
 - 对于那些梯度值落在两个给定阈值范围之内的像素他们被标记为弱边缘（也就是作为最终边缘图的候选者）； 
 - 如果弱边缘与强边缘相连，他们最终会保留在边缘图中。
 
 
**2.2 黑色琴键识别**

矩形筛选首先使用cv.findContours函数来查找上面边缘检测结果图片的轮廓：
![在这里插入图片描述](https://img-blog.csdnimg.cn/975e005b3c134157929ffa4cb4de4fc2.png)

    这里轮廓检索模式使用的是RETR_TREE模式，即检索所有的轮廓，并重构嵌套轮廓的整个层次。这种模式下会返回所有轮廓，并且创建一个完整的组织结构列表。它甚至会告诉你谁是爷爷，爸爸，儿子，孙子等。以右图为例，使用这种模式，对OpenCV返回的结果重新排序并分析它，红色数字是边界的序号，绿色是组织结构轮廓0的组织结构为0，同一级中Next为7，没有Previous。子轮廓是1，没有父轮廓。所以数组是[7，-1，1，-1]。轮廓1的组织结构为1，同一级中没有其他，没有Previous。子轮廓是2，父轮廓为0。所以数组是[-1，-1，2，0]。轮廓逼近方式选择的是CHAIN_APPROX_SIMPLE模式，意为压缩水平的、垂直的和斜的部分，也就是函数只保留他们的终点部分。检测出轮廓后，先进行面积筛选：
 ![在这里插入图片描述](https://img-blog.csdnimg.cn/8d8dba39f1424b12a45d416e0f2011b2.png)

    进行轮廓的外接矩形的筛选，使用minAreaRect函数提取外界矩阵，得到外界矩阵的中心点及面积，使用以下四个指标：
（1）contour要逼近外接矩形的面积
（2）长宽比正确
（3）筛除过于靠近的contour
（4）不能和已有的contour重合度过高
 
 

**2.3 白色琴键识别**
通过以上识别到10个黑色按键的外接矩形，需要通过其寻找到白键。首先获取黑键下方的点。minAreaRect函数返回的rect对象，rect[0]返回矩形的中心点，rect[1]返回矩形的长和宽，可以由此通过邻接矩阵长边与短边返回
 ![在这里插入图片描述](https://img-blog.csdnimg.cn/64e2ffeaf9b349cdb5559e7bce539183.png)



