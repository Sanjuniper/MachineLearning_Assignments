import numpy as np
import pandas as pd
import torch
import torch.nn as nn

#1.获取数据

def getData(filename):
    data = pd.read_csv(filename)
    #修改数据类型
    data = data.astype('int')
    train_data = np.array(data.iloc[:2800,1:58])
    train_label = np.array(data.iloc[:2800,58])
    test_data = np.array(data.iloc[2800:, 1:58])
    test_label = np.array(data.iloc[2800:, 58])
    return train_data,train_label,test_data,test_label

#数据归一化
def normalization(data):
    min_val = data.min(0)
    max_val = data.max(0)
    #字符数字转化为数字


    ranges = max_val-min_val
    num = ranges.shape[0]
    for i in range(num):
        if ranges[i] == 0 :
            ranges[i] = 1

    norm_data = (data-min_val)/ranges
    return norm_data
#2.构建模型
def Sigmoid(x):
    return 1/(1+np.exp(-x))

def ClassificationModel(w,x,b):
    return Sigmoid(np.dot(w,x)+b)

#3.训练
def train(traindata,trainlabel,epoch):
    lr=8
    train_num = traindata.shape[0]
    dim = traindata.shape[1]
    b=0
    w = np.ones(dim) #返回全是1的1*dim矩阵

    w_sum=0
    b_sum = 0
    reg_rate = 0.001

    for i in range(epoch):
        gram_w = np.zeros(dim)
        gram_b = 0
        acc = 0
        for j in range(train_num):
            x = traindata[j]
            t = trainlabel[j]
            y = ClassificationModel(w,x,b)
            if(y>0.5):
                y1 = 1
            else:
                y1 = 0

            if(y1 == t):
                acc=acc+1


            #梯度下降，本次实验由于loss值太大，因此采用公式计算梯度
            gram_b = gram_b+(-1) * (t - y)
            for k in range(dim):
                gram_w[k] = gram_w[k] +(-1) * (t - y) * x[k] + 2 * reg_rate * w[k]

        gram_b = gram_b/train_num
        gram_w = gram_w/train_num
        #注意在计算总和的时候计算平方
        w_sum = w_sum+(gram_w**2)
        b_sum = b_sum+(gram_b**2)
        train_acc = acc/train_num

        #Adagrad 梯度下降
        b -= (lr ) * gram_b
        w -= lr  * gram_w

        if i%5 ==0 :
            print('第',i,'轮:','准确率:',train_acc)

    return w,b
def test(testdata,testlabel,w,b):
    testnum = testdata.shape[0]
    acc = 0
    for i in range(testnum):
        x = testdata[i]
        t = testlabel[i]
        y = ClassificationModel(w,x,b)
        if y>0.5:
            y1 = 1
        else:
            y1 = 0
        if y1==t:
            acc = acc+1

    return acc/testnum


traindata,trainlabel,testdata,testlable = getData('spam_train.csv')
traindata = normalization(traindata)
w,b = train(traindata,trainlabel,50)
testdata = normalization(testdata)
print('测试准确率:',test(testdata,testlable,w,b))









