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
