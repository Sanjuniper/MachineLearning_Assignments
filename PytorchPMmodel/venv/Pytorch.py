import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader

#导入数据集
#Pytorch实现双层神经网络

class PmDataset(Dataset):
    def __init__(self):
        data = pd.read_csv('PMtrain.csv')
        data = data.iloc[:3044, :14]
        data = data.set_index(['測項'])
        data = data.loc[['PM2.5']]
        self.y = np.array(data['10'])
        self.y = self.y.astype('double')
        self.y = torch.as_tensor(torch.from_numpy(self.y), dtype=torch.float32)
        data = data.iloc[:, 3:12]
        self.x = np.array(data)
        self.x = self.x.astype('double')
        self.x = torch.as_tensor(torch.from_numpy(self.x), dtype=torch.float32)


    def __getitem__(self, item):
        return self._x[item],self.y_[item]
    def __len__(self):
        return len(self.y)



#pytorch搭建网络

PMnet= nn.Sequential(
    nn.Linear(9,81),
    nn.ReLU(),
    nn.Linear(81,729),
    nn.ReLU(),
    nn.Linear(729,81),
    nn.ReLU(),
    nn.Linear(81,9),
    nn.ReLU(),
    nn.Linear(9,1)
)

traindata = PmDataset()
num_train = traindata.__len__()


#定义损失函数和优化器
loss_func = nn.MSELoss()
optimzer = torch.optim.SGD(PMnet.parameters(), lr=0.0001)


#训练
def train(traindata,epoch):

    for i in range(epoch):
        trainloss = 0
        for j in range(num_train):
            y = PMnet(traindata.x[j])
            y = y.squeeze(-1)
            yh = traindata.y[j]
            yh = yh.squeeze(-1)

            loss = loss_func(y,yh)

            trainloss += loss
            #清零梯度
            optimzer.zero_grad()
            loss.backward()
            optimzer.step()

        trainloss = trainloss/num_train
        if i%10 == 0:
            print('epoch:',i,' loss:',trainloss/2)

    torch.save(PMnet, 'PMnet.pth')


train(traindata,200)
checkpoint = torch.load('PMnet.pth')
print(checkpoint(traindata.x[0]),traindata.y[0])
for j in range(num_train):
    if j%10 ==0:
        print(checkpoint(traindata.x[j]), traindata.y[j])


'''
yh = traindata.y[0]
yh =  yh.squeeze(-1)
y= torch.tensor([40],dtype=torch.float32)
yh =  yh.squeeze(-1)
print(loss_func(y,yh))
'''







