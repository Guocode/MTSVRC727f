#!/usr/bin/python
# -*- coding: utf-8 -*-
#every video contributes to 64 frames
import cv2
import numpy as np
import matplotlib.pyplot as plt
import datetime
'''
function: return RGB & Flow [outframe*reh*rew*3]
'''
def cvread_video_rgb(video_path,outframe=16,reh=256,rew=256,show=False):
    cap = cv2.VideoCapture(video_path)
    frames_num = cap.get(7)
    fps = cap.get(5)
    RGBs = []
    frames = [round((frames_num-1)/(outframe-1)*x) for x in list(range(outframe))] #frame index to read
    prev = np.zeros((reh,rew,3),dtype='uint8')
    if show:
        print("fps:",fps,"frames_num",frames_num)
        print(frames)
    if cap.isOpened():
        for i in frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)  # 设置要获取的帧号
            ret, frame = cap.read()
            if ret:
                frame = cv2.resize(frame, (reh, rew), interpolation=cv2.INTER_CUBIC)
            else:
                frame = prev #读取失败 应该循环尝试读取上一帧  cap.set(cv2.CAP_PROP_POS_FRAMES, i-1)
            RGBs.append(frame)
            prev = frame
            # if show:
            #     mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            #     hsv[..., 0] = ang * 180 / np.pi / 2
            #     hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
            #     rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            #     cv2.imshow("frame", frame)
            #     cv2.imshow("flow", rgb)
            #     cv2.imshow("frame0", frame)
            # k = cv2.waitKey(20)
            # #q q键退出
            # if (k & 0xff == ord('q')):
            #     break

    cap.release()
    RGBs = np.array(RGBs)
    return RGBs

def cvread_video_flow(video_path,outframe=16,reh=256,rew=256,show=False):
    prev = np.zeros((reh,rew,3),dtype='uint8')
    hsv = np.zeros((256,256,3),dtype='uint8')
    hsv[..., 1] = 255
    flows = []
    cap = cv2.VideoCapture(video_path)
    frames_num = cap.get(7)
    fps = cap.get(5)
    num = 0
    frames = [round((frames_num-1)/(outframe-1)*x) for x in list(range(outframe))] #frame index to read
    if show:
        print("fps:",fps,"frames_num",frames_num)
        print(frames)
    if cap.isOpened():
        for i in frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)  # 设置要获取的帧号
            ret, frame = cap.read()
            if i != 0:
                # flow = cv2.calcOpticalFlowFarneback(prev, cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), None, 0.7, 9, 7, 3, 5, 1.2, 0)
                flow = cv2.calcOpticalFlowFarneback(cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY), cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), None, 0.5, 3, 6,
                                                    3, 5, 1.25, 0)
                flows.append(flow)
            else:
                prev = frame
                continue
            prev = frame
            # if show:
            #     mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            #     hsv[..., 0] = ang * 180 / np.pi / 2
            #     hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
            #     rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            #     cv2.imshow("frame", frame)
            #     cv2.imshow("flow", rgb)
            #     cv2.imshow("frame0", frame)
            # k = cv2.waitKey(20)
            # #q q键退出
            # if (k & 0xff == ord('q')):
            #     break

    cap.release()
    flows = np.array(flows)
    return flows

if __name__ == '__main__':
    d1 = datetime.datetime.now()
    path = "D:/1000006114.mp4" #"/Users/guoziheng/Movies/5.mp4"
    rgbs = cvread_video_rgb(path,outframe=64,reh=256,rew=256,show=True)
    np.save("np.npy",rgbs) #每512个视频一读 就是6G内存
    d2 = datetime.datetime.now()
    print((d2-d1))
    print(rgbs.shape)

    d1 = datetime.datetime.now()
    a = np.load("np.npy")
    d2 = datetime.datetime.now()
    print((d2-d1))