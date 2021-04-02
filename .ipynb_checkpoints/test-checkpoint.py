# Application of FL task
from MLModel import *
from FLModel import *
from utils import *

from torchvision import datasets, transforms
import torch
import numpy as np
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = "cpu"

def load_cnn_mnist(num_users):
    data_train = datasets.MNIST(root="./data/", train=True, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]))

    data_test = datasets.MNIST(root="./data/", train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]))

    # split MNIST (training set) into non-iid data sets
    non_iid = []
    user_dict = mnist_noniid(data_train, num_users)
    for i in range(num_users):
        print("i",i)
        idx = user_dict[i]
        #print("idx",idx)
        d = data_train.data[idx].float().unsqueeze(1)
        print("D",len(d[0]))
        print("D type",d.shape)  #30000, 1, 28, 28
        targets = data_train.targets[idx].float()
        print("targets",targets)
        #print("(d, targets)",(d, targets))
        non_iid.append((d, targets))
    non_iid.append((data_test.data.float().unsqueeze(1), data_test.targets.float()))
    return non_iid
import random
import csv
import pandas
def load_cnn_virus(num_users):


    dataframe = pandas.read_csv('0_300.csv')

    array = dataframe.values
    random.shuffle(array) # random the dataset

    features = array[:,0:300]

    labels = array[:,300] 
    #labels = (labels <14)
    from sklearn import preprocessing

    min_max_scaler = preprocessing.MinMaxScaler()
    features = min_max_scaler.fit_transform(features)
    features = torch.FloatTensor(features)
    from sklearn.model_selection import train_test_split

    train_features,test_features, train_labels, test_labels = train_test_split(features,labels,  test_size = 0.2, random_state = 0)  

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

    non_iid.append((test_features,test_labels))
    return non_iid
#print(test)
client_num = 3
d = load_cnn_virus(client_num)
print("len d",len(d))#d = client_num+1 add 1 is test dataset
print("d 0",d[2])
lr =0.0001
fl_param = {
    'output_size': 15,
    'client_num': client_num,
    'model': MLP,
    'data': d,
    'lr': lr,
    'E': 1,
    'C': 1,
    #'sigma': 0.5,
    'epsilon':0.5,
    'delta':0.5,
    'clip': 4,
    'batch_size': 256,
    'device': device
}
import warnings
warnings.filterwarnings("ignore")
fl_entity = FLServer(fl_param).to(device)
import asyncio

print("mnist")
max_acc = -1
for e in range(2000):
    '''
    if (e+1) % 30 == 0:
        print("hahahah")
        lr *= 0.1
        fl_entity.set_lr(lr)
    '''
    acc = asyncio.run(fl_entity.global_update())
    
    if acc>max_acc:
        max_acc = acc
        lr_max = lr
    print("global epochs = {:d}, acc = {:.4f},max_acc = {:.4f},lr = {:.10f},lr_max = {:.10f}".format(e+1, acc,max_acc,lr,lr_max ))
