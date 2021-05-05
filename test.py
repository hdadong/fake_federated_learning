# Application of FL task
from MLModel import *
from FLModel import *
from utils import *

from torchvision import datasets, transforms
import torch
import numpy as np
import os

import warnings
warnings.filterwarnings("ignore")
import asyncio

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = "cpu"


import random
import csv
import pandas
#读取iid数据
def load_cnn_virus(ifMAL,ifbinary,num_users):
    """
    ifmal:为1时使用mal数据集，0为使用anl数据集
    ifbinary:为1时使用二分类，0时多酚类
    num_users:为客户端数量
    
    """

    if ifMAL == 0 and ifbinary==0 :
        dataframe = pandas.read_csv("0_300.csv")
        array = dataframe.values
        random.shuffle(array) # random the dataset
        features = array[:,0:300]
        labels = array[:,300]
    elif (ifMAL==0 and ifbinary == 1):
        dataframe = pandas.read_csv("0_300.csv")
        array = dataframe.values
        random.shuffle(array) # random the dataset
        features = array[:,0:300]
        labels = array[:,300]
        labels = labels<14
    elif ifMAL == 1 and ifbinary == 0:
        dataframe = pandas.read_csv("feature_vectors_syscallsbinders_frequency_5_Cat.csv")
        array = dataframe.values
        random.shuffle(array) # random the dataset
        features = array[:,0:470]
        labels = array[:,470]-1
    elif ifMAL == 1 and ifbinary == 1:
        dataframe = pandas.read_csv("feature_vectors_syscallsbinders_frequency_5_Cat.csv")
        array = dataframe.values
        #划分1795个恶心和良性
        bad = array[0:9804,:]
        random.shuffle(bad) # random the dataset
        bad = bad[0:1795,:]
        
        good = array[9804:,:]
        bad_good = np.vstack((bad,good))
        random.shuffle(bad_good)
        
        features = bad_good[:,0:470]
        labels = bad_good[:,470]-1
        labels = labels<4
 
    from sklearn import preprocessing
    min_max_scaler = preprocessing.MinMaxScaler()
    features = min_max_scaler.fit_transform(features)
    features = torch.FloatTensor(features)
    from sklearn.model_selection import train_test_split

    train_features,test_features, train_labels, test_labels = train_test_split(features,labels,  test_size = 0.2, random_state = 1)  

    size = int(len(train_features)/num_users)

    non_iid = []

    for i in range(num_users):
        if i != num_users - 1:
            #print(train_data[0:10])
            train_f = train_features[i*size:(i+1)*(size)]#11598, 1, 47, 10
            train_l = train_labels[i*size:(i+1)*(size)]
            non_iid.append((train_f,train_l))
            print("train_f",train_f.shape)
        if i == num_users - 1:
            train_f = train_features[i*size:]
            train_l = train_labels[i*size:]
            non_iid.append((train_f,train_l))
            print("train_f",train_f.shape)
    '''
    for i in range(len(test_labels)):
        print(test_labels[i])
    '''
    non_iid.append((test_features,test_labels))
    return non_iid
#数据数量倾斜，读取数据
def notavg_load_cnn_virus(ifMAL,ifbinary):
    """
    ifmal:为1时使用mal数据集，0为使用anl数据集
    ifbinary:为1时使用二分类，0时多酚类
    num_users:为客户端数量
    
    """

    if ifMAL == 0 and ifbinary==0 :
        dataframe = pandas.read_csv("0_300.csv")
        array = dataframe.values
        random.shuffle(array) # random the dataset
        features = array[:,0:300]
        labels = array[:,300]
    elif (ifMAL==0 and ifbinary == 1):
        dataframe = pandas.read_csv("0_300.csv")
        array = dataframe.values
        random.shuffle(array) # random the dataset
        features = array[:,0:300]
        labels = array[:,300]
        labels = labels<14
    elif ifMAL == 1 and ifbinary == 0:
        dataframe = pandas.read_csv("feature_vectors_syscallsbinders_frequency_5_Cat.csv")
        array = dataframe.values
        random.shuffle(array) # random the dataset
        features = array[:,0:470]
        labels = array[:,470]-1
    elif ifMAL == 1 and ifbinary == 1:
        dataframe = pandas.read_csv("feature_vectors_syscallsbinders_frequency_5_Cat.csv")
        array = dataframe.values
        #划分1795个恶心和良性
        bad = array[0:9804,:]
        random.shuffle(bad) # random the dataset
        bad = bad[0:1795,:]
        
        good = array[9804:,:]
        bad_good = np.vstack((bad,good))
        random.shuffle(bad_good)
        
        features = bad_good[:,0:470]
        labels = bad_good[:,470]-1
        labels = labels<4

        

        
    from sklearn import preprocessing

    min_max_scaler = preprocessing.MinMaxScaler()
    features = min_max_scaler.fit_transform(features)
    features = torch.FloatTensor(features)
    from sklearn.model_selection import train_test_split

    train_features,test_features, train_labels, test_labels = train_test_split(features,labels,  test_size = 0.1, random_state = 1)  

    size = int(len(features)/18) # 划分成20份，2份测试

    non_iid = []


    
    list1 = [1,1,2,3,5,6] #划分成6份，这是每份的占比
    list2 = [0,1,2,4,7,12,18] #list的前n项和
    for i in range(len(list1)):
        if i != len(list1) - 1:

            train_f = train_features[list2[i]*size:list2[i+1]*(size)]#11598, 1, 47, 10
            train_l = train_labels[list2[i]*size:list2[i+1]*(size)]
            non_iid.append((train_f,train_l))
            print("train_f",train_f.shape)
        if i == len(list1) - 1:
            train_f = train_features[list2[i]*size:]
            train_l = train_labels[list2[i]*size:]
            non_iid.append((train_f,train_l))
            print("train_f",train_f.shape)

    non_iid.append((test_features,test_labels))
    return non_iid

#数据标签倾斜读取数据
def labelnotavg_load_cnn_virus(ifMAL,ifbinary,num_users,noniid_label):

    """
    ifmal:为1时使用mal数据集，0为使用anl数据集
    ifbinary:为1时使用二分类，0时多酚类
    num_users:为客户端数量
    noniid_label：为每个客户端分配的标签类别的数量。
    """

    if ifMAL == 0 and ifbinary==0 :
        dataframe = pandas.read_csv("0_300.csv")
        array = dataframe.values
        random.shuffle(array) # random the dataset
        features = array[:,0:300]
        labels = array[:,300]
        num_label = 15
        num_feature = 300
    elif ifMAL == 1 and ifbinary == 0:
        dataframe = pandas.read_csv("feature_vectors_syscallsbinders_frequency_5_Cat.csv")
        array = dataframe.values
        random.shuffle(array) # random the dataset
        features = array[:,0:470]
        labels = array[:,470]-1
        num_label = 5
        num_feature = 470
   
    from sklearn import preprocessing
    min_max_scaler = preprocessing.MinMaxScaler()
    features = min_max_scaler.fit_transform(features)
    features = torch.FloatTensor(features)
    labels = torch.LongTensor(labels)
    from sklearn.model_selection import train_test_split
    train_features,test_features, train_labels, test_labels = train_test_split(features,labels,  test_size = 0.1, random_state = 1)  


    size = int(len(train_features)/num_users)

    non_iid = []

    for i in range(num_users):
        keep_labels = np.random.choice(range(num_label),noniid_label, replace=False)
        indices = np.isin(train_labels , keep_labels).astype("uint8")
        train_f = (torch.masked_select(train_features.transpose(0, 1), torch.tensor(indices)).view(num_feature, -1).transpose(1, 0)).to(device)
        train_l = torch.masked_select(train_labels, torch.tensor(indices)).to(device)
        non_iid.append((train_f,train_l))
        print("train_f",train_f.shape)

    non_iid.append((test_features,test_labels))
    return non_iid



client_num = 50  #设置客户端数量
aggregate_round = 200 #设置聚合次数T
noniid_label = 2    #设置每个客户端有多少个类别


maxaccs = []

for p in range(0,3):#做三次实验
    #读取标签倾斜的mal 5分类数据集
    d = labelnotavg_load_cnn_virus(1,0,client_num,noniid_label)

    accs=[]

    fl_param = {
    'noise':False,#是否在参数上添加噪音
    'noise2':False,#是否在梯度上添加噪音
    'ifclip':False,#是否进行参数剪切，如果要加噪音则要剪切
    'ifclip2':False,#是否进行梯度剪切，如果要加噪音则要剪切
    'whichoptimizer':2,#4:avg-adam，3:save-SGD，2:savd-adam，1:nonsave-adam，0:nonsave-SGD
    'output_size': 5,#预测类被为5个类
    'client_num':client_num,#客户端数量
    'model': MLP,#DNN神经网络
    'data': d,
    'lr': 0.0001,
    'E': 1,#本地客户端的训练epoch

    'C': 0.3,#客户端每轮的参与率
    'method':1,#0表示fedavg,1表示逆距离

    'epsilon':0,#参数的隐私预算，不添加噪音默认为0
    'delta':0,#参数的松弛项，不添加噪音默认为0
    'epsilon2': 0,#梯度的隐私预算，不添加噪音默认为0
    'delta2':0,#梯度的松弛项，不添加噪音默认为0
    'clip':  0,#参数的剪切阈值，不添加噪音默认为0
    'clip2':0,#梯度的剪切阈值，不添加噪音默认为0

    'batch_size': 256,
    'device': device#设置为GPU或CPU
    }


    fl_entity = FLServer(fl_param).to(device)
    maxacc=-1
    for e in range(aggregate_round):
        t = []
        acc = asyncio.run(fl_entity.global_update())
        if acc>maxacc:
            maxacc=acc
        print("第{}次迭代,acc:{},maxacc:{}".format(e,acc,maxacc))

        t.append(acc)
    #print(t)
    maxaccs.append(maxacc)
print("each experience acc:{}".format(maxaccs))
print("noniid_label:{},avg_Acc:{}".format(noniid_label,np.sum(maxaccs[:])/3))    