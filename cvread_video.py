#!/usr/bin/python
# -*- coding: utf-8 -*-
#every video contributes to 64 frames
import cv2
import numpy as np
import matplotlib.pyplot as plt
'''
function: return RGB & Flow [outframe*reh*rew*3]
'''
def cvread_video(video_path,outframe=16,reh=256,rew=256,show=False):
    cap = cv2.VideoCapture(video_path)
    frames_num = cap.get(7)
    # interval = frames_num // 64
    fps = cap.get(5)
    num = 0
    if show:
        print("fps:",fps,"frames_num",frames_num)
    RGBs = []
    flows = []
    frames = [round((frames_num-1)/(outframe-1)*x) for x in list(range(outframe))]
    if show:
        print(frames)
    prev = np.zeros((reh,rew,3),dtype='uint8')
    hsv = np.zeros((256,256,3),dtype='uint8')
    hsv[..., 1] = 255
    while cap.isOpened():
        for i in frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)  # 设置要获取的帧号
            ret, frame = cap.read()
            if ret:
                frame = cv2.resize(frame, (reh, rew), interpolation=cv2.INTER_CUBIC)
            else:
                frame = prev
            RGBs.append(frame)
            if i!=0:
                # flow = cv2.calcOpticalFlowFarneback(prev, cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), None, 0.7, 9, 7, 3, 5, 1.2, 0)
                flow = cv2.calcOpticalFlowFarneback(cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY), cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), None, 0.5, 3, 6,
                                                    3, 5, 1.25, 0)
                flows.append(flow)
            else:
                prev = frame
                continue
            prev = frame

            if show:
                mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                hsv[..., 0] = ang * 180 / np.pi / 2
                hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
                rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
                cv2.imshow("frame", frame)
                cv2.imshow("flow", rgb)
                cv2.imshow("frame0", frame)

            # k = cv2.waitKey(20)
            # #q q键退出
            # if (k & 0xff == ord('q')):
            #     break
        break

    cap.release()
    RGBs = np.array(RGBs)
    flows = np.array(flows)
    return RGBs,flows

if __name__ == '__main__':
    rgbs,flows = cvread_video("/media/guo/搬砖BOY/dataset/840202103.mp4",outframe=16,reh=256,rew=256,show=True)
    print(rgbs.shape,flows.shape)