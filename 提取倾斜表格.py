#coding:utf-8
'''
表格生成线条坐标
'''
import cv2
import numpy as np
from matplotlib import pyplot as plt
import json
import sys
import subprocess
import os


def detectTable(imgpath):
    img1 = cv2.imread(imgpath)
    #img1 =cv2.resize(img1,None,fx=0.3, fy=0.3, interpolation = cv2.INTER_CUBIC)
    if len(img1.shape) == 2:  # 灰度图
        gray_img = img1
    elif len(img1.shape) ==3:
        gray_img = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

    thresh_img = cv2.adaptiveThreshold(~gray_img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,15,-2)
    h_img = thresh_img.copy()
    v_img = thresh_img.copy()
    
    '''
    此处形态学因子是一条直线，长度一定要设置的合适，由于表格有倾斜，
    图像越大，该值就要设置的越大，否则不能提取出水平线       
    '''
    h_structure = cv2.getStructuringElement(cv2.MORPH_RECT,(30,1)) # 形态学因子
    h_erode_img = cv2.erode(h_img,h_structure,1)

    #膨胀图像
    h_dilate_img = cv2.dilate(h_erode_img,h_structure,1)
    # cv2.imshow("h_erode",h_dilate_img)

    
    #此处形态学因子是一条直线，长度一定要设置的合适，否则不能提取出垂直线
    v_structure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 30))  # 形态学因子
    v_erode_img = cv2.erode(v_img, v_structure, 1)
     #膨胀图像
    v_dilate_img = cv2.dilate(v_erode_img, v_structure, 1)

    mask_img = h_dilate_img+v_dilate_img
    win = cv2.namedWindow("mask", flags=2)
    cv2.imshow("mask",mask_img)

    #cv2.bitwise_and图像进行与运算
    joints_img = cv2.bitwise_and(h_dilate_img,v_dilate_img)
    win = cv2.namedWindow("joints", flags=2)
    cv2.imshow("joints",joints_img)    

    binary,contours,hierarchy= cv2.findContours(mask_img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)  
    print(type(contours),len(contours))#共几个轮廓


    '''第一个参数是指明在哪幅图像上绘制轮廓；
    第二个参数是轮廓本身，在Python中是一个list。
    第三个参数指定绘制轮廓list中的哪条轮廓，如果是-1，则绘制其中的所有轮廓。
    第四个参数表示颜色
    第五个参数表示轮廓线的宽度，如果是-1，则为填充模式'''
    img3 = img1.copy()
    cv2.drawContours(img3,contours,-1,(0,0,255),2)  


    #cv2.contourArea计算轮廓面积,返回轮廓内像素点的个数，此处将轮廓集按面积排序
    c = sorted(contours, key=cv2.contourArea, reverse=True)[0]

    #cv2.minAreaRect主要求得包含点集最小面积的矩形，这个矩形是可以有偏转角度的，可以与图像的边界不平行。
    rect = cv2.minAreaRect(c)
    print(type(rect),rect)
    box = np.int0(cv2.boxPoints(rect))
    cv2.drawContours(img3, [box], -1, (0, 255, 0), 10)
    win = cv2.namedWindow("largest shape", flags=2)
    cv2.imshow("largest shape",img3)
    cv2.waitKey(0)  
    cv2.destroyAllWindows()

    Xs = [i[0] for i in box]
    Ys = [i[1] for i in box]
    x1 = min(Xs)
    x2 = max(Xs)
    y1 = min(Ys)
    y2 = max(Ys)
    hight = y2 - y1
    width = x2 - x1
    x3=int(x1+width*0.72)
    y3=int(y1+hight*0.02)
    x4=x2
    y4=int(y2-hight*0.91)
    print(x1,y1)
    print(x2,y2)
    print(x3,y3)
    #cropImg = img[y1:y2, x1:x2]
    cropImg = img1[y3:y4, x3:x4]
    cv2.imshow("crop shape",cropImg)
    cv2.waitKey(0)  
    cv2.destroyAllWindows()

a = r'p0.png'
detectTable(a)

