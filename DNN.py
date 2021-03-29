import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets,transforms
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader, TensorDataset
##import pandas
#training setting
batch_size = 16
# MNIST Dataset
import random
import csv
import pandas
###########import numpy
################import pandas sb4
def load_cnn_virus(num_users):


    dataframe = pandas.read_csv('feature_vectors_syscallsbinders_frequency_5_Cat.csv')

    array = dataframe.values
    random.shuffle(array) # random the dataset
    features = array[:,0:470]
    size = int(len(features)/num_users)
    labels = array[:,470] - 1


    features = torch.FloatTensor(features)
    print("features222",features.shape)

    print(len(features[0]))
    features = features.reshape(-1,470) #transfer to a image
    labels = torch.LongTensor(labels)


    print("features",features.shape) # #11598, 1, 47, 10
    train_features = features[0:9000] #select 9000 as training set
    test_features = features[9000:] #other as testing set


    train_labels = labels[0:9000] #select 9000 as training set
    test_labels = labels[9000:] #other as testing set
    

    non_iid = []

    for i in range(num_users):
        if i != num_users - 1:
            #print(train_data[0:10])
            train_f = train_features[i*size:(i+1)*size]
            print("train_f",train_f.shape)
            train_l = train_labels[i*size:(i+1)*size]
            non_iid.append((train_f,train_l))
        if i == num_users - 1:
            train_f = train_features[i*size:]
            train_l = train_labels[i*size:]
            non_iid.append((train_f,train_l))

    non_iid.append((test_features,test_labels))
    return non_iid
#print(test)
client_num = 1
d = load_cnn_virus(client_num)
# Data Loader (Input Pipeline)

data=d[0]
print("d0",torch.tensor(data[0]).shape)
print("d1",torch.tensor(data[1]).shape)
torch_dataset = TensorDataset(torch.tensor(data[0]),torch.tensor(data[1]))

train_loader = torch.utils.data.DataLoader(dataset=torch_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)
data=d[1]
torch_dataset = TensorDataset(torch.tensor(data[0]),torch.tensor(data[1]))
test_loader = torch.utils.data.DataLoader(dataset=torch_dataset,
                                           batch_size=batch_size,
                                           shuffle=False)






class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.l1 = nn.Linear(470,520)
        self.l2 = nn.Linear(520, 320)
        #self.l3 = nn.Linear(320, 240)
        #self.l4 = nn.Linear(240, 120)
        self.l5 = nn.Linear(320, 5)

    def forward(self, x):
        
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        #x = F.relu(self.l3(x))
        #x = F.relu(self.l4(x))

        return self.l5(x)
model = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(),lr= 0.001 , momentum= 0.5)

def train(epoch):
    model.train()
    for data1, target in train_loader:
        print("data1",data1)
        data1,target = Variable(data1), Variable(target)
        optimizer.zero_grad()
        output = model(data1)
        print("output",output)
        print("target",target)
        loss = criterion(output,target)
        loss.backward()
        optimizer.step()
        '''
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
        '''
    for batch_idx, (data1, target) in enumerate(train_loader):
        print("data1")
        data1,target = Variable(data1), Variable(target)
        optimizer.zero_grad()
        output = model(data1)
        print("output",output)
        print("target",target)
        loss = criterion(output,target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


def test():
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data, target = Variable(data,volatile=True),Variable(target)
        output = model(data)
        # sum up batch loss
        test_loss += criterion(output, target).data.item()
        # get the index of the max
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

for epoch in range(1, 10):
    train(epoch)
    test()
