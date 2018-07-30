import numpy as np
import pandas as pd
import threading
from p3d_model import P3D199, P3D63
import torch
from cvread_video import cvread_video_rgb
from torch.utils.data import TensorDataset, DataLoader
import os

device = 'cuda' if torch.cuda.is_available() else 'cpu'
import logging
import os.path
import time

# 第一步，创建一个logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)  # Log等级总开关
# 第二步，创建一个handler，用于写入日志文件
rq = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
log_path = os.path.dirname(os.getcwd()) + '/Logs/'
log_name = './Logs/' + rq + '.log'
print("log file in:", log_name)
logfile = log_name
fh = logging.FileHandler(logfile, mode='w')
fh.setLevel(logging.DEBUG)  # 输出到file的log等级的开关
# 第三步，定义handler的输出格式
formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
fh.setFormatter(formatter)
# 第四步，将logger添加到handler里面
logger.addHandler(fh)


class MTSVRC:
    def __init__(self, data_path, label_train_file, label_val_file):
        self.data_size = None
        self.data_path = data_path  # "/media/guo/搬砖BOY/dataset/"#"D:/dataset/"#/media/guo/搬砖BOY/dataset/"
        self.label_train = pd.read_csv('./train_label.cvs', header=None,
                                       sep=',')  # self.load_label(label_train_file,shuffle=True)#("/media/guo/搬砖BOY/English/trainEnglish.txt")#("D:/dataset/English/trainEnglish.txt")
        self.label_val = pd.read_csv('./val_label.cvs', header=None,
                                     sep=',')  # self.load_label(label_val_file,shuffle=False)#("/media/guo/搬砖BOY/English/valEnglish.txt")#("D:/dataset/English/valEnglish.txt")
        self.p3d_model = P3D63(num_classes=50)
        self.ch_en_labels = {'宠物狗': 'dog',
                             '宠物猫': 'cat',
                             '宠物鼠': 'rat',
                             '宠物兔子': 'rabbit',
                             '宠物鸟': 'bird',
                             '风景': 'scenery',
                             '风土人情': 'customs',
                             '穿秀': 'clothes showing',
                             '宝宝': 'child',
                             '男生自拍': 'boy selfie',
                             '女生自拍': 'girl selfie',
                             '做甜品': 'dessert',
                             '做海鲜': 'seafood',
                             '做小吃': 'snack',
                             '饮品': 'drinks',
                             '抓娃娃': 'doll catching',
                             '手势舞': 'finger dance',
                             '街舞': 'street dance',
                             '国标舞': 'Ballroom dance',
                             '钢管舞': 'pole dance',
                             '芭蕾舞': 'ballet',
                             '绘画': 'painting',
                             '手写文字': 'handwriting',
                             '咖啡拉花': 'coffee art',
                             '沙画': 'sand art',
                             '史莱姆': 'slime',
                             '折纸': 'origami',
                             '编织': 'weave',
                             '陶艺': 'ceramic art',
                             '手机壳': 'phone shell',
                             '打鼓': 'drum playing',
                             '弹吉他': 'guitar playing',
                             '弹钢琴': 'piano playing',
                             '弹古筝': 'Zheng playing',
                             '拉小提琴': 'violin',
                             '唱歌': 'singing',
                             '游戏': 'game playing',
                             '动漫': 'cartoon',
                             '瑜伽': 'yoga',
                             '健身': 'fitness',
                             '滑板': 'skateboard',
                             '篮球': 'basketball playing',
                             '跑酷': 'parkour',
                             '潜水': 'diving',
                             '台球': 'billiards',
                             '画眉': 'brow makeup',
                             '画眼': 'eye makeup',
                             '唇彩': 'lips makeup',
                             '美甲': 'manicure',
                             '美发': 'hairdressing'
                             }
        self.labels = list(self.ch_en_labels.values())
        # self.label_train.loc[:, 2] = self.label_train.loc[:, 2].replace(dict(zip(self.labels, list(range(50)))))
        # self.label_val.loc[:, 2] = self.label_val.loc[:, 2].replace(dict(zip(self.labels, list(range(50)))))
        # self.label_train.to_csv('train_label.cvs', index=False, header=False)
        # self.label_val.to_csv('val_label.cvs', index=False, header=False)

    def load_data(self):
        pass

    def train(self, filein_batch=16, batch_size=8, epoch_num=5, learning_rate=0.01, save=False, train_num=64701):
        print("Start training!")
        self.p3d_model.to(device)
        for epoch in range(epoch_num):  # loop over the dataset multiple times
            print('\nStart Epoch: %d' % (epoch + 1))
            sample_index = 0
            epoch_corr = 0
            epoch_train_loss = 0
            epoch_batch_idx = 0
            global preload_train_data
            preload_train_data = None
            global preload_label_data
            preload_label_data = None
            global preloaded
            preloaded = False
            while (sample_index + filein_batch <= train_num):  # file batch
                if preloaded:  # start a loaddata thread and join the thread
                    # print("preloaded!")
                    traindata = preload_train_data
                    labeldata = preload_label_data
                else:
                    loaddata_thread = loaddata_Thread(sample_index, filein_batch, self)
                    loaddata_thread.start()
                    loaddata_thread.join()
                    traindata = preload_train_data
                    labeldata = preload_label_data
                    print(traindata.shape, labeldata.shape)
                preloaded = False
                '''
                    create a thread to preload data 
                '''
                loaddata_thread = loaddata_Thread(sample_index + filein_batch, filein_batch, self)
                loaddata_thread.start()
                ############# data loaded, start training

                dataloader = DataLoader(TensorDataset(traindata, labeldata), batch_size=batch_size, shuffle=False)
                for batch_idx, (inputs_batch, targets_batch) in enumerate(dataloader):  # data batch
                    '''
                        train on batch with p3d model
                    '''
                    correct, train_loss = self.p3d_model.train_on_batch(inputs_batch, targets_batch,
                                                                        batch_size=batch_size,
                                                                        learning_rate=learning_rate, save=save,
                                                                        curepoch=epoch)
                    epoch_corr += correct
                    epoch_train_loss += train_loss
                    sample_index += batch_size
                    epoch_batch_idx += 1
                    acc = epoch_corr / sample_index
                    step_info = ('Epoch: %d | %d/%d | ' % (epoch + 1, sample_index, train_num) +
                                 'Loss: %.3f | Acc: %.3f%% (%d/%d) | Batch_Acc: %.3f%% (%d/%d) | Batch_Loss: %.3f'
                                 % (epoch_train_loss / epoch_batch_idx, 100. * acc, epoch_corr, sample_index,
                                    100. * correct / batch_size, correct, batch_size, train_loss))
                    print('Epoch: %d | %d/%d | ' % (epoch + 1, sample_index, train_num),
                          'Loss: %.3f | Acc: %.3f%% (%d/%d) | Batch_Acc: %.3f%% (%d/%d) | Batch_Loss: %.3f'
                          % (epoch_train_loss / epoch_batch_idx, 100. * acc, epoch_corr, sample_index,
                             100. * correct / batch_size, correct, batch_size, train_loss))

                    logger.info(step_info)

                if not preloaded:
                    loaddata_thread.join()
            if save:
                self.save(acc, epoch)
        print('Training has finished!')

    def val(self, filein_batch=128, batch_size=16):
        print("Start validating!")
        self.p3d_model.to(device)
        saved_model = "./checkpoint/ckpt-epoch-5.t7"
        if os._exists(saved_model):
            weights = torch.load(saved_model)['state_dict']
            self.p3d_model.load_state_dict(weights)
            print("weight loaded!")
        else:
            print("weight file not found!")
        train_num = 16192
        for epoch in range(1):  # loop over the dataset multiple times
            print('\nStart Epoch: %d' % (epoch + 1))
            sample_index = 0
            corr = 0
            val_loss = 0
            global preload_train_data
            preload_train_data = None
            global preload_label_data
            preload_label_data = None
            global preloaded
            preloaded = False
            while (sample_index + filein_batch <= train_num):  # file batch
                if preloaded:  # start a loaddata thread and join the thread
                    # print("preloaded!")
                    traindata = preload_train_data
                    labeldata = preload_label_data
                else:
                    loaddata_thread = loaddata_Thread(sample_index, filein_batch, self)
                    loaddata_thread.start()
                    loaddata_thread.join()
                    traindata = preload_train_data
                    labeldata = preload_label_data
                    print(traindata.shape, labeldata.shape)
                preloaded = False
                '''
                    create a thread to preload data 
                '''
                loaddata_thread = loaddata_Thread(sample_index + filein_batch, filein_batch, self)
                loaddata_thread.start()
                ############# data loaded, start training

                dataloader = DataLoader(TensorDataset(traindata, labeldata), batch_size=batch_size, shuffle=False)
                for batch_idx, (inputs_batch, targets_batch) in enumerate(dataloader):  # data batch
                    '''
                        validate on batch with p3d model
                    '''
                    _correct, _val_loss = self.p3d_model.val_model(inputs_batch, targets_batch)
                    corr += _correct
                    val_loss += _val_loss
                    sample_index += batch_size
                    batch_idx += 1
                    acc = corr / sample_index
                    print('Index: %d/%d | ' % (sample_index, train_num),
                          'Loss: %.3f | Acc: %.3f%% (%d/%d)' % (
                              val_loss / sample_index, 100. * acc, corr, sample_index))
                if not preloaded:
                    loaddata_thread.join()
                del (loaddata_thread)

    def save(self, acc, epoch):
        self.p3d_model.save_model(acc, epoch)
        print('Model has been saved!')

    def load_label(self, txt_file_path, shuffle=True):
        if shuffle:
            return pd.read_table(txt_file_path, header=None, encoding='utf-8', delimiter=',').sample(frac=1)
        else:
            return pd.read_table(txt_file_path, header=None, encoding='utf-8', delimiter=',')

    def load_labelnames(self, txt_file_path):
        return pd.read_table(txt_file_path, header=None, encoding='utf-8', delimiter=':,')

    def data_check(self, filein_batch=16, epoch_num=1):  # run without training to check data
        print("Start checking!")
        train_num = 64701
        for epoch in range(epoch_num):  # loop over the dataset multiple times
            print('\nStart Epoch: %d' % (epoch + 1))
            sample_index = 0
            global preload_train_data
            preload_train_data = None
            global preload_label_data
            preload_label_data = None
            global preloaded
            preloaded = False
            while (sample_index + filein_batch <= train_num):  # file batch
                if preloaded:  # start a loaddata thread and join the thread
                    # print("preloaded!")
                    traindata = preload_train_data
                    labeldata = preload_label_data
                else:
                    loaddata_thread = loaddata_Thread(sample_index, filein_batch, self)
                    loaddata_thread.start()
                    loaddata_thread.join()
                    traindata = preload_train_data
                    labeldata = preload_label_data
                    print(traindata.shape, labeldata.shape)
                preloaded = False
                '''
                    create a thread to preload data 
                '''
                loaddata_thread = loaddata_Thread(sample_index + filein_batch, filein_batch, self)
                loaddata_thread.start()
                sample_index += filein_batch
                if not preloaded:
                    loaddata_thread.join()
                print('sample_index', sample_index, '/', train_num)

        print('Data check passed!')


class loaddata_Thread(threading.Thread):  # 继承父类threading.Thread
    def __init__(self, sample_index, filein_batch, mtsvrc):
        threading.Thread.__init__(self)
        self.sample_index = sample_index
        self.filein_batch = filein_batch
        self.mtsvrc = mtsvrc

    def run(self):  # 把要执行的代码写到run函数里面 线程在创建后会直接运行run函数
        traindata = []
        for filename in mtsvrc.label_train.iloc[self.sample_index:self.sample_index + self.filein_batch, 0]:
            rgb = cvread_video_rgb(mtsvrc.data_path + str(filename) + '.mp4', outframe=16, reh=256, rew=256)
            traindata.append(rgb)
        traindata = np.array(traindata).reshape(self.filein_batch, 3, 16, 256, 256) / 256
        traindata = torch.from_numpy(traindata)  # Variable(torch.rand(1, 3, 16, 256, 256))
        labeldata = torch.from_numpy(
            mtsvrc.label_train.iloc[self.sample_index:self.sample_index + self.filein_batch, 2].values)
        global preload_train_data
        global preload_label_data
        global preloaded
        preload_train_data = traindata
        preload_label_data = labeldata
        preloaded = True


if __name__ == '__main__':
    data_path = "/media/guo/搬砖BOY/dataset/"  # "D:/dataset/"#/media/guo/搬砖BOY/dataset/"
    label_train = "/media/guo/搬砖BOY/English/trainEnglish.txt"  # ("D:/dataset/English/trainEnglish.txt")
    label_val = "/media/guo/搬砖BOY/English/valEnglish.txt"  # ("D:/dataset/English/valEnglish.txt")
    mtsvrc = MTSVRC(data_path, label_train, label_val)
    print("cuda:" + str(torch.cuda.is_available()))
    # mtsvrc.val(filein_batch = 2,batch_size = 2)
    # mtsvrc.data_check(filein_batch=8)
    mtsvrc.train(batch_size=1, filein_batch=2)
    # print(mtsvrc.label_val.head())
    print("exit")
