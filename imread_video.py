#!/usr/bin/python
# -*- coding: utf-8 -*-
#every video contributes to 64 frames
import numpy as np
import cv2

cap = cv2.VideoCapture('XXX.mp4')
wid = int(cap.get(3))
hei = int(cap.get(4))
framerate = int(cap.get(5))
framenum = int(cap.get(7))

video = np.zeros((framenum, hei, wid, 3), dtype='float16')
cap.set(cv2.CAP_PROP_POS_FRAMES,50)  #设置要获取的帧号

cnt = 0
while (cap.isOpened()):
    a, b = cap.read()
    cv2.imshow('%d' % cnt, b)
    cv2.waitKey(20)
    b = b.astype('float16') / 255
    video[cnt] = b
    print(cnt)
    cnt += 1