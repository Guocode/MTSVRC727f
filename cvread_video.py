#!/usr/bin/python
# -*- coding: utf-8 -*-
# every video contributes to 16 frames
import cv2
import numpy as np
import matplotlib.pyplot as plt
import datetime

'''
function: return RGB & Flow [outframe*reh*rew*3]
'''


def cv_read_video_iframe(cap, i):  # 若读取失败，循环读取上一帧
    if i < 0:
        return np.zeros((256, 256, 3), dtype='uint8')
    cap.set(cv2.CAP_PROP_POS_FRAMES, i)  # 设置要获取的帧号
    ret, frame = cap.read()
    if ret:
        return frame
    else:
        return cv_read_video_iframe(cap, i - 2)


def cv_read_video_i2frame(cap, i):  # 若读取失败，循环读取上一帧
    if i < 0:
        return np.zeros((256, 256, 3), dtype='uint8'), np.zeros((256, 256, 3), dtype='uint8')
    cap.set(cv2.CAP_PROP_POS_FRAMES, i)  # 设置要获取的帧号
    ret1, frame1 = cap.read()
    cap.set(cv2.CAP_PROP_POS_FRAMES, i + 1)  # 设置要获取的帧号
    ret2, frame2 = cap.read()
    if ret1 and ret2:
        return frame1, frame2
    else:
        return cv_read_video_i2frame(cap, i - 2)


def cvread_video_rgb(video_path, outframe=16, reh=256, rew=256, show=False):
    cap = cv2.VideoCapture(video_path)
    frames_num = cap.get(7)
    fps = cap.get(5)
    rgbs = []
    frames = [round((frames_num - 1) / (outframe - 1) * x) for x in list(range(outframe))]  # frame index to read
    # prev = np.zeros((reh,rew,3),dtype='uint8')
    if show:
        print("fps:", fps, "frames_num", frames_num)
        print(frames)
    if cap.isOpened():
        for i in frames:
            try:
                frame = cv_read_video_iframe(cap, i)
                frame = cv2.resize(frame, (reh, rew), interpolation=cv2.INTER_CUBIC)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgbs.append(frame)
            except Exception as e:
                print(e)
                print(video_path, " : ", i, " no passed")
            # prev = frame
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
    else:
        raise Exception("cannot open video of ", video_path)
    cap.release()
    return np.array(rgbs)


def cvread_video_flow(video_path, outframe=16, reh=256, rew=256, show=False):
    prev = np.zeros((reh, rew, 3), dtype='uint8')
    hsv = np.zeros((256, 256, 3), dtype='uint8')
    hsv[..., 1] = 255
    flows = []
    cap = cv2.VideoCapture(video_path)
    frames_num = cap.get(7)
    fps = cap.get(5)
    frames = [round((frames_num - 2) / (outframe - 1) * x) for x in list(range(outframe))]  # frame index to read
    if show:
        print("fps:", fps, "frames_num", frames_num)
        print(frames)
    if cap.isOpened():
        for i in frames:
            try:
                frame1, frame2 = cv_read_video_i2frame(cap, i)
                frame1 = cv2.resize(frame1, (reh, rew), interpolation=cv2.INTER_CUBIC)
                frame2 = cv2.resize(frame2, (reh, rew), interpolation=cv2.INTER_CUBIC)
                # flow = cv2.calcOpticalFlowFarneback(prev, cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), None, 0.7, 9, 7, 3, 5, 1.2, 0)
                flow = cv2.calcOpticalFlowFarneback(cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY),
                                                    cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY), None, 0.5, 3, 6,
                                                    3, 5, 1.25, 0)
                flows.append(flow)
            except Exception as e:
                print(e)
                print(video_path, " : ", i, " no passed")

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
    else:
        raise Exception("cannot open video of ", video_path)
    cap.release()
    return np.array(flows)


if __name__ == '__main__':
    path = "/media/guo/搬砖BOY/dataset/911706308.mp4"  # "/Users/guoziheng/Movies/5.mp4"
    d1 = datetime.datetime.now()
    flows = cvread_video_flow(path, outframe=16, reh=256, rew=256, show=True)
    d2 = datetime.datetime.now()
    rgbs = cvread_video_rgb(path, outframe=16, reh=256, rew=256, show=True)
    d3 = datetime.datetime.now()
    print(d2 - d1)
    print(d3 - d2)
