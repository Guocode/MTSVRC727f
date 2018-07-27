import numpy as np
import pandas as pd
from p3d_model import P3D199,P3D63
import torch
from cvread_video import cvread_video
from torch.utils.data import TensorDataset, DataLoader
device = 'cuda' if torch.cuda.is_available() else 'cpu'
class MTSVRC:
    def __init__(self,data_path,label_train_file,label_val_file):
        self.data_size = None
        self.data_path = data_path#"/media/guo/搬砖BOY/dataset/"#"D:/dataset/"#/media/guo/搬砖BOY/dataset/"
        self.label_train = self.load_label(label_train_file,shuffle=True)#("/media/guo/搬砖BOY/English/trainEnglish.txt")#("D:/dataset/English/trainEnglish.txt")
        self.label_val = self.load_label(label_val_file)#("/media/guo/搬砖BOY/English/valEnglish.txt")#("D:/dataset/English/valEnglish.txt")
        self.p3d_model = P3D199(num_classes=50)
        self.ch_en_labels ={'宠物狗':'dog',
                            '宠物猫':'cat',
                            '宠物鼠':'rat',
                            '宠物兔子':'rabbit',
                            '宠物鸟':'bird',
                            '风景':'scenery',
                            '风土人情':'customs',
                            '穿秀':'clothes showing',
                            '宝宝':'child',
                            '男生自拍':'boy selfie',
                            '女生自拍':'girl selfie',
                            '做甜品':'dessert',
                            '做海鲜':'seafood',
                            '做小吃':'snack',
                            '饮品':'drinks',
                            '抓娃娃':'doll catching',
                            '手势舞':'finger dance',
                            '街舞':'street dance',
                            '国标舞':'Ballroom dance',
                            '钢管舞':'pole dance',
                            '芭蕾舞':'ballet',
                            '绘画':'painting',
                            '手写文字':'handwriting',
                            '咖啡拉花':'coffee art',
                            '沙画':'sand art',
                            '史莱姆':'slime',
                            '折纸':'origami',
                            '编织':'weave',
                            '陶艺':'ceramic art',
                            '手机壳':'phone shell',
                            '打鼓':'drum playing',
                            '弹吉他':'guitar playing',
                            '弹钢琴':'piano playing',
                            '弹古筝':'Zheng playing',
                            '拉小提琴':'violin',
                            '唱歌':'singing',
                            '游戏':'game playing',
                            '动漫':'cartoon',
                            '瑜伽':'yoga',
                            '健身':'fitness',
                            '滑板':'skateboard',
                            '篮球':'basketball playing',
                            '跑酷':'parkour',
                            '潜水':'diving',
                            '台球':'billiards',
                            '画眉':'brow makeup',
                            '画眼':'eye makeup',
                            '唇彩':'lips makeup',
                            '美甲':'manicure',
                            '美发':'hairdressing'
                            }
        self.labels = list(self.ch_en_labels.values())
        self.label_train.loc[:, 2] = self.label_train.loc[:, 2].replace(dict(zip(self.labels, list(range(50)))))
        self.label_val.loc[:, 2] = self.label_val.loc[:, 2].replace(dict(zip(self.labels, list(range(50)))))

    def load_data(self):
        pass
    def train(self,filein_batch = 16,batch_size=8,epoch_num=5, learning_rate=0.01, save=False):
        print("Start training!")
        self.p3d_model.to(device)
        train_num = 64701
        for epoch in range(epoch_num):  # loop over the dataset multiple times
            print('\nStart Epoch: %d' % (epoch + 1))
            sample_index = 0
            epoch_corr = 0
            epoch_train_loss = 0
            epoch_batch_idx = 1
            while(sample_index+filein_batch<=train_num):#file batch
                traindata = []
                for filename in self.label_train.iloc[sample_index:sample_index+filein_batch,0]:
                    rgb, flows = cvread_video(self.data_path+str(filename)+'.mp4',outframe=16,reh=256,rew=256)
                    traindata.append(rgb)
                traindata = np.array(traindata).reshape(filein_batch,3,16,256,256)/256
                traindata = torch.from_numpy(traindata)#Variable(torch.rand(1, 3, 16, 256, 256))
                labeldata = torch.from_numpy(self.label_train.iloc[sample_index:sample_index+filein_batch,2].values)
                if(sample_index==0):
                    print(traindata.shape,labeldata.shape)
                dataloader = DataLoader(TensorDataset(traindata, labeldata), batch_size=batch_size, shuffle=False)
                for batch_idx, (inputs_batch, targets_batch) in enumerate(dataloader):#data batch
                    '''
                        train on batch with p3d model
                    '''
                    correct, train_loss = self.p3d_model.train_on_batch(inputs_batch, targets_batch, batch_size=batch_size,
                                                            learning_rate=learning_rate, save=save, curepoch=epoch)
                    epoch_corr += correct
                    epoch_train_loss += train_loss
                    sample_index += batch_size
                    epoch_batch_idx += 1
                    acc = epoch_corr / sample_index
                    print('Epoch: %d | %d/%d | ' % (epoch + 1, sample_index, train_num),
                          'Loss: %.3f | Acc: %.3f%% (%d/%d) | Batch_Acc: %.3f%% (%d/%d)'
                          % (epoch_train_loss / epoch_batch_idx, 100. * acc, epoch_corr, sample_index,100. * correct / batch_size, correct, batch_size))
            if save:
                self.save(acc,epoch)
        print('Training has finished!')
        self.p3d_model.val_model()
    def save(self,acc,epoch):
        self.p3d_model.save_model(acc,epoch)
        print('Model has been saved!')
    def load_label(self,txt_file_path,shuffle=True):
        if shuffle:
            return pd.read_table(txt_file_path,header=None,encoding='utf-8',delimiter=',').sample(frac = 1)
        else:
            return pd.read_table(txt_file_path,header=None,encoding='utf-8',delimiter=',')
    def load_labelnames(self,txt_file_path):
        return pd.read_table(txt_file_path, header=None, encoding='utf-8', delimiter=':,')
    def data_check(self, filein_batch=16, batch_size=8, epoch_num=1, learning_rate=0.01, save=False): #run without training to check data
        print("Start checking!")
        train_num = 60000
        for epoch in range(epoch_num):  # loop over the dataset multiple times
            print('\nStart Epoch: %d' % (epoch + 1))
            sample_index = 0
            while(sample_index+filein_batch<=train_num):#file batch
                traindata = []
                for filename in self.label_train.loc[sample_index:sample_index+filein_batch-1,0]:
                    rgb, flows = cvread_video(self.data_path+str(filename)+'.mp4',outframe=16,reh=256,rew=256)
                    traindata.append(rgb)
                traindata = np.array(traindata).reshape(filein_batch,3,16,256,256)/256
                traindata = torch.from_numpy(traindata)#Variable(torch.rand(1, 3, 16, 256, 256))
                labeldata = torch.from_numpy(self.label_train.loc[sample_index:sample_index+filein_batch-1,2].values)
                sample_index += filein_batch
                print(traindata.shape,labeldata.shape)
                print('sample_index',sample_index,'/',train_num)
        print('Data check passed!')
if __name__ == '__main__':
    data_path = "/opt/dataset/"#"/media/guo/搬砖BOY/dataset/"#"D:/dataset/"#/media/guo/搬砖BOY/dataset/"
    label_train = "./trainEnglish.txt"#"/media/guo/搬砖BOY/English/trainEnglish.txt"#("D:/dataset/English/trainEnglish.txt")
    label_val = "./valEnglish.txt"#("D:/dataset/English/valEnglish.txt")
    mtsvrc = MTSVRC(data_path,label_train,label_val)
    print("cuda:"+str(torch.cuda.is_available()))
    #mtsvrc.data_check(batch_size=8,filein_batch=64,learning_rate=0.05)
    mtsvrc.train(batch_size=8,filein_batch=64,learning_rate=0.05)
    # print(mtsvrc.label_val.head())
    print("exit")