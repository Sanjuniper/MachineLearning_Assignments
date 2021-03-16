import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
from torchvision import transforms, datasets, models
from PIL import Image
import numpy as np
from torch.autograd import Variable
from torch import optim, cuda
import matplotlib.pyplot as plt
#导入数据,建立数据集


traindir = 'training'
testdir = 'validation'
train_on_gpu = cuda.is_available()
print('Train on gpu: {}'.format(train_on_gpu))
batchsize = 128


class FoodDataset(Dataset):
    def __init__(self):
        self.images,self.labels = self.read_file(testdir)

    def read_file(self, path):
        #读取path内所有文件名
        labels = []
        files_list = os.listdir(path)
        for i in range(len(files_list)):
            labels.append(files_list[i].split('_')[0])
        #path加上所有文件名
        file_path_list = [os.path.join(path, img) for img in files_list]
        labels = list(map(int, labels))
        return file_path_list,labels

    def __getitem__(self, item):
        img = self.images[item]
        img = Image.open(img)
        img = self.img_transform(img)
        label = self.labels[item]
        label = torch.tensor(label)
        return img,label

    def __len__(self):
        return len(self.images)


    def img_transform(self,img):
        transform = transforms.Compose(
            [
                transforms.Resize((227, 227)),  # 将输入的PIL图片转换成给定的尺寸的大小
                transforms.ToTensor(),  # 将PIL图片或者numpy.ndarray转成Tensor类型的
                transforms.Normalize([0.5, 0.5, 0.5],  # 用均值和标准差对张量图像进行标准化处理
                                     [0.5, 0.5, 0.5])  # Imagenet standards
            ]
        )
        img = transform(img)
        return img


m =nn.Softmax(dim=1)

#利用Torch搭建卷积神经网络
class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.conv1 = nn.Sequential( #inputsize 227*227*3
            nn.Conv2d(in_channels=3,out_channels=48,kernel_size=11,stride=4,padding=0),
            nn.ReLU(),
            nn.MaxPool2d(3,2) #outputsize 27*27*48
        )
        self.conv2 = nn.Sequential(#inputsize 27*27*48
            nn.Conv2d(48,128,5,1,2),
            nn.ReLU(),
            nn.MaxPool2d(3,2)  #outputsize 13*13*128
        )
        '''
        self.conv3 = nn.Sequential(  # inputsize 13*13*128
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.ReLU(), # outputsize 13*13*256
        )
        '''
        self.conv3 = nn.Sequential(  # inputsize 13*13*128
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.ReLU(),  # outputsize 13*13*256
            nn.MaxPool2d(3,2)  # outputsize 6*6*128
        )
        self.fclayer= nn.Sequential(
            nn.Linear(4608,1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512,11)
        )

    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
       # x = self.conv4(x)
        x = x.view(x.size(0),-1) #flatten
        x = self.fclayer(x)
        return m(x)


#建立训练过程



def train(dataloader,epoch,loss,optimizer,data_num):
    train_loss = []
    train_acc = []
    print('The training process has began:')
    for i in range(epoch):
        trainloss=0.0
        trainacc=0
        pred = []
        for data,datalabel in dataloader:
            if train_on_gpu:
                data,datalabel = data.cuda(),datalabel.cuda()
            y = model.forward(data)
            optimizer.zero_grad()
            loss1 = loss(y,datalabel)
            trainloss += loss1*y.size(0)
            loss1.backward()
            optimizer.step()
            pred = torch.max(y,dim=1)
            pred = pred.indices
            for k in range(len(data)):
                if pred[k] == datalabel[k]:
                    trainacc=trainacc+1

        trainloss = trainloss/data_num
        trainacc = trainacc/data_num
        train_loss.append(trainloss)
        train_acc.append(trainacc)



        print('epoch',i+1,'trainloss:',trainloss,'trainacc:',trainacc)

    torch.save(model, 'Alexnet-food11.pth')

    return train_loss,train_acc

def test(dataloader,data_num):
    testacc=0
    for data,datalabel in dataloader:
        if train_on_gpu:
            data, datalabel = data.cuda(), datalabel.cuda()
        y = model.forward(data)
        pred = torch.max(y, dim=1)
        pred = pred.indices
        for k in range(len(data)):
            if pred[k] == datalabel[k]:
                testacc = testacc + 1




    print(testacc)

    testacc = testacc/data_num
    return testacc


#model = AlexNet()
model = torch.load('Alexnet-food11.pth')
model = model.cuda()
loss = nn.CrossEntropyLoss() #定义损失函数
optimizer = optim.Adam(model.parameters(),lr=0.0001)
data = FoodDataset()
dataloader = DataLoader(data,batch_size=batchsize,shuffle=True,drop_last=True)

data_num = len(data)
print(data_num)


testacc = test(dataloader,data_num)
print(testacc)


#tl,ta=train(dataloader,25,loss,optimizer,data_num)




#绘图
'''

x = np.arange(0,15,1) #以1为单位生成0-35
plt.plot(x,tl,label="trainloss")
plt.xlabel("epoch")
plt.ylabel("trainloss")
plt.title('loss && epoch')
plt.legend()
plt.show()
plt.xlabel("epoch")
plt.ylabel("trainacc")
plt.plot(x,ta,label="trainloss")
plt.title('acc && epoch')
plt.show()

'''
'''
data = FoodDataset()
dataloader = DataLoader(data,batch_size=64,shuffle=True,drop_last=False)
'''


'''
data,label = dataset.__getitem__(1000)
data = Variable(torch.unsqueeze(data, dim=0).float(), requires_grad=False)
model = AlexNet()
y = model.forward(data)
t = label
loss = nn.CrossEntropyLoss()

'''

