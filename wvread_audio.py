import numpy as np
from  pydub import AudioSegment
import pywt
import matplotlib.pyplot as plt
def load_video(file_path):
    audio = AudioSegment.from_file(file_path, 'mp4')
    waveData = np.fromstring(audio.raw_data, dtype=np.int16)#np.fromstring(strData, dtype=np.int16)  # 将字符串转化为int
    nchannels = audio.channels
    framerate = audio.frame_rate
    nframes = waveData.shape[0] // nchannels
    pltframe = 1024
    time = np.arange(0, pltframe) * (1.0 / framerate)
    waveData = waveData * 1.0 / (max(abs(waveData)))  # wave幅值归一化
    waveData = np.reshape(waveData, [nframes, nchannels])
    np_audio = waveData[:, 0] #first channel
    np_audio = np_audio[88200:88200 + 1024]

    # 傅里叶变换
    # transformed = np.fft.fft(np_audio)  # 傅里叶变换
    # shifted = np.fft.fftshift(transformed)  # 移频
    # plt.figure()
    # plt.subplot(2,1,1)
    # plt.plot(time, np_audio)
    # plt.subplot(2,1,2)
    # plt.plot(transformed)  # 绘制变换后的信号
    # plt.show()

    #每1ms有44.1frame 每960ms有42336frame
    wavename = 'cgau8'
    totalscal = framerate // 4
    fc = pywt.central_frequency(wavename)
    cparam = 2 * fc * totalscal
    scales = cparam / np.arange(totalscal, 1, -1)
    [cwtmatr, frequencies] = pywt.cwt(np_audio, scales, wavename, 1.0 / framerate)
    plt.figure(figsize=(8, 4))
    plt.subplot(211)
    plt.plot(time, np_audio)
    plt.xlabel(u"time(s)")
    plt.title(u"300Hz 200Hz 100Hz Time spectrum")
    plt.subplot(212)
    plt.contourf(time, frequencies, abs(cwtmatr))
    plt.ylabel(u"freq(Hz)")
    plt.xlabel(u"time(s)")
    plt.subplots_adjust(hspace=0.4)
    plt.show()

if __name__ == '__main__':
    load_video("D:/dataset/group0/843302365.mp4")
