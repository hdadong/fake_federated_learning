# Federated Learning Model in PyTorch
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from DPMechanisms import gaussian_noise_weight,gaussian_noise, clip_weight,gaussian_noise_ls, clip_grad

import numpy as np
import copy
import random


class FLClient(nn.Module):
    """ Client of Federated Learning framework.
        1. Receive global model from server
        2. Perform local training (compute gradients)
        3. Return local model (gradients) to server
    """
    def __init__(self,noise, noise2,ifclip,ifclip2,whichoptimizer,model, output_size, data, lr, E, batch_size, clip, clip2,epsilon,delta,epsilon2,delta2, device=None):
    #def __init__(self, model, output_size, data, lr, E, batch_size, clip, sigma, device=None):
        """
        :param model: ML model's training process should be implemented
        :param data: (tuple) dataset, all data in client side is used as training data
        :param lr: learning rate
        :param E: epoch of local update
        """
        super(FLClient, self).__init__()
        self.noise=noise
        self.noise2=noise2


        self.ifclip =  ifclip
        self.ifclip2 =  ifclip2
        self.whichoptimizer = whichoptimizer
        self.device = device
        self.BATCH_SIZE = batch_size
        torch_dataset = TensorDataset(torch.tensor(data[0]),
                                      torch.tensor(data[1]))
        self.data_size = len(torch_dataset)
        self.data_loader = DataLoader(
            dataset=torch_dataset,
            batch_size=self.BATCH_SIZE,
            shuffle=True
        )

        self.lr = lr
        self.E = E
        self.clip = clip
        self.clip2 = clip2

        #self.sigma = sigma
        self.epsilon = epsilon
        self.delta = delta
        self.epsilon2 = epsilon2
        self.delta2 = delta2
        print()
        print("data[0].shape[1]",data[0].shape[1])
        self.model = model(data[0].shape[1], output_size).to(self.device)
        self.batch_model = model(data[0].shape[1], output_size).to(self.device)
        self.recv_model = model(data[0].shape[1], output_size).to(self.device)
        if self.whichoptimizer == 2:
            self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.lr)
        elif self.whichoptimizer == 3:
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9)
        elif self.whichoptimizer == 4:
            self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.lr)
    def recv(self, model_par,opt_par):
        """receive global model from aggregator (server)"""
        self.model.load_state_dict(copy.deepcopy(model_par))
        if self.whichoptimizer == 4:
            self.optimizer.load_state_dict(copy.deepcopy(opt_par))

    async def update(self):
        """local model update"""
        self.model.train()
        criterion = nn.CrossEntropyLoss()
        if self.whichoptimizer==0:
            optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9)
        if self.whichoptimizer== 1:
            optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.lr)
        if self.whichoptimizer==2:
            optimizer = self.optimizer
            #print(optimizer.state_dict)
            #print(optimizer.state_dict.param_groups[1])
            #print(optimizer.param_groups)
            #print((optimizer.param_groups))
            

                #print(group['params'])
        if self.whichoptimizer==3:
            optimizer = self.optimizer
        if self.whichoptimizer==4:
            optimizer = self.optimizer
        t = 0
        for e in range(self.E):
            for batch_x, batch_y in self.data_loader:
                #print("batch_x",batch_x)
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                pred_y = self.model(batch_x)
                #print("pred_y",pred_y)
                loss = criterion(pred_y, batch_y.long()) / len(self.data_loader)
                loss.backward()
                # bound l2 sensitivity (gradient clipping)
                
                if self.ifclip == True:
                    grads = dict(self.model.named_parameters())
                    for name in grads:
                        grads[name].grad = clip_grad(grads[name].grad, self.clip)
                        if self.noise2 == True:
                            sensitivity2 = 2 *  self.clip / self.BATCH_SIZE 
                            grads[name].grad += gaussian_noise_weight(grads[name].grad.shape, sensitivity2,self.epsilon2, self.delta2, device=self.device)
                    self.model.load_state_dict(grads)
                
                        #print(grads[name].grad)
                        
                optimizer.step()
                optimizer.zero_grad()
                if self.ifclip2 == True:
                    weight = dict(self.model.named_parameters())
                    for name in weight:
                        #print(name)
                        weight[name] = clip_weight(weight[name], self.clip2)
                        self.model.load_state_dict(weight)
                        
                        #print(weight[name])


                '''
                torch.save(optimizer.state_dict(), "./optimizer.pt")
                optimizer = torch.optim.Adam(params=self.model.parameters(), lr=0.1)
                optimizer.load_state_dict(torch.load('./optimizer.pt'))
                
                for group in optimizer.param_groups:
   
                    for p in group['params']:
                        state = optimizer.state[p]
                        print(state['step'])
                        #print(state['exp_avg'] )
                        #print(state['exp_avg_sq'])
                    #print(len(group['params']))
                    #print(group['params'])
                    #optimizer.zero_grad()
                    print("hahhahaha")
                    #for i in range(0,6):
                        #print(group['params'][i])
                
                t=t+1
                if t>60:
                    break
                '''
                
        # Add Gaussian noise
        # 1. compute l2-sensitivity by Client Based DP-FedAVG Alg.
        # 2. add noise
        if self.noise == True:
            #print("noise!!!")
            sensitivity = 2 *  self.clip2 / self.BATCH_SIZE 
            new_param = copy.deepcopy(self.model.state_dict())
            for name in new_param:
                new_param[name] = torch.zeros(new_param[name].shape).to(self.device)
                new_param[name] += 1.0 * self.model.state_dict()[name]
                new_param[name] += gaussian_noise_weight(self.model.state_dict()[name].shape, sensitivity,self.epsilon, self.delta, device=self.device)
            self.model.load_state_dict(copy.deepcopy(new_param))
        '''
        if self.noise2 == True:   
            sensitivity2 = 2 *  self.clip / self.BATCH_SIZE 
            new_opt_par =  copy.deepcopy(self.optimizer.state_dict())
            for index in new_opt_par["param_groups"][0][ 'params']:
                new_opt_par["state"][index]['exp_avg'] = new_opt_par["state"][index]['exp_avg']+gaussian_noise_weight(new_opt_par["state"][index]['exp_avg'] .shape, sensitivity2,self.epsilon2, self.delta2, device=self.device)
            self.optimizer.load_state_dict(copy.deepcopy(new_opt_par))

        if self.noise3 == True:  
            sensitivity3 = 2 *  self.clip*  self.clip / self.BATCH_SIZE 
            new_opt_par =  copy.deepcopy(self.optimizer.state_dict())

            for index in new_opt_par["param_groups"][0][ 'params']:
                #print(gaussian_noise_weight(new_opt_par["state"][index]['exp_avg'] .shape, sensitivity3,self.epsilon3, self.delta3, device=self.device))
                new_opt_par["state"][index]['exp_avg_sq'] = new_opt_par["state"][index]['exp_avg_sq'] + gaussian_noise_weight(new_opt_par["state"][index]['exp_avg'] .shape, sensitivity3,self.epsilon3, self.delta3, device=self.device)
            #self.optimizer.load_state_dict(copy.deepcopy(new_opt_par))
        '''
        return 1
        
        

class FLServer(nn.Module):
    """ Server of Federated Learning
        1. Receive model (or gradients) from clients
        2. Aggregate local models (or gradients)
        3. Compute global model, broadcast global model to clients
    """
    def __init__(self, fl_par):
        super(FLServer, self).__init__()
        self.noise = fl_par['noise']
        self.noise2 = fl_par['noise2']

        self.ifclip =  fl_par['ifclip']
        self.whichoptimizer =  fl_par['whichoptimizer']
        self.device = fl_par['device']
        self.client_num = fl_par['client_num']
        self.C = fl_par['C']  # (float) C in [0, 1]
        self.method = fl_par['method']
        self.clip = fl_par['clip']
        self.clip = fl_par['clip2']

        #self.data = torch.tensor(fl_par['data'][-1][0]).to(self.device)  # test set
        #self.target = torch.tensor(fl_par['data'][-1][1]).to(self.device)  # target label

        # For FEMnist dataset
        self.data = []
        self.target = []
        for sample in fl_par['data'][self.client_num:]:
            self.data += [torch.tensor(sample[0]).to(self.device)]  # test set
            self.target += [torch.tensor(sample[1]).to(self.device)]  # target label

        self.input_size = int(self.data[0].shape[1])
        self.lr = fl_par['lr']
        print("self.input_size",self.input_size)
        self.clients = [FLClient(fl_par['noise'],
                                 fl_par['noise2'],
                                 fl_par['ifclip'],
                                 fl_par['ifclip2'],
                                 fl_par['whichoptimizer'],
                                 fl_par['model'],
                                 fl_par['output_size'],
                                 fl_par['data'][i],
                                 fl_par['lr'],
                                 fl_par['E'],
                                 fl_par['batch_size'],
                                 fl_par['clip'],
                                 fl_par['clip2'],
                                 #fl_par['sigma'],
                                 fl_par['epsilon'],
                                 fl_par['delta'],
                                 #fl_par['sigma'],
                                 fl_par['epsilon2'],
                                 fl_par['delta2'],
                                 self.device)
                        for i in range(self.client_num)]
        self.global_model = fl_par['model'](self.input_size, fl_par['output_size']).to(self.device)

        self.weight = np.array([client.data_size * 1.0 for client in self.clients])
        if self.whichoptimizer == 4:
            self.global_optimizer = torch.optim.Adam(params=self.global_model.parameters(), lr=self.lr)
            self.broadcast(self.global_model.state_dict(),self.global_optimizer.state_dict())

        elif self.whichoptimizer <4 :
            self.broadcast(self.global_model.state_dict(),self.global_model.state_dict())
    def aggregated(self, idxs_users):
        """FedAvg"""
        model_par = [self.clients[idx].model.state_dict() for idx in idxs_users]
        ifidd = self.method

        new_par = copy.deepcopy(model_par[0])


        for name in new_par:
            #print("fafafa")
            #print(new_par[name])
            new_par[name] = torch.zeros(new_par[name].shape).to(self.device)
        #fedavg
        if ifidd == 0:

            for idx, par in enumerate(model_par):
                #print(idxs_users[:])

                p_i = self.weight[idxs_users[idx]] / np.sum(self.weight[idxs_users[:]])


                for name in new_par:
                    new_par[name] += par[name] * (p_i)
        #逆距离
        elif ifidd == 1:
            for idx, par in enumerate(model_par):
                #w_Avg
                for name in new_par:
                    new_par[name] += par[name] /len(idxs_users[:])
            w_Avg = copy.deepcopy(new_par)
            w_Avg_w = copy.deepcopy(new_par)
            Z = copy.deepcopy(new_par)

            for name in new_par:
                new_par[name] = torch.zeros(new_par[name].shape).to(self.device)
                w_Avg_w[name] = torch.zeros(new_par[name].shape).to(self.device)
                Z[name] = torch.zeros([]).to(self.device)
 
            for idx, par in enumerate(model_par):
                #w_avg - w
                for name in new_par:
                    w_Avg_w[name] = w_Avg[name] - par[name]
                    w_Avg_w[name].flatten()
                    w_Avg_w[name] = torch.norm(w_Avg_w[name], p=1)
                    Z[name] += 1/w_Avg_w[name]

            for idx, par in enumerate(model_par):
                for name in new_par:
                    new_par[name] += (1/torch.norm((w_Avg[name]-par[name]).flatten(),p=1))*(1/Z[name])*par[name]
                
        self.global_model.load_state_dict(copy.deepcopy(new_par))
        
        #优化器参数聚合
        if self.whichoptimizer == 4:
            opt_par =  [self.clients[idx].optimizer.state_dict() for idx in idxs_users] 
            new_opt_par = copy.deepcopy(opt_par[0])
            #print(new_opt_par["state"]) #这是每层神经网络的参数:step,vt,mt,是一个Dict
            #print(new_opt_par["param_groups"][0][ 'params'])#这是模型每一层参数的索引，是一个list
            #print(new_opt_par["param_groups"][0])#这是模型每一层参数的索引，是一个list,state param_groups
            for index in new_opt_par["param_groups"][0][ 'params']:

                new_opt_par["state"][index]['step'] = 0
                new_opt_par["state"][index]['exp_avg'] = torch.zeros(new_opt_par["state"][index]['exp_avg'].shape).to(self.device)
                new_opt_par["state"][index]['exp_avg_sq'] = torch.zeros(new_opt_par["state"][index]['exp_avg_sq'].shape).to(self.device)
                #print(new_opt_par["state"][index]) #这是index这一层神经网络的参数:step,vt,mt,是一个Dict

            for idx, par in enumerate(opt_par):
                p_i = self.weight[idxs_users[idx]] / np.sum(self.weight[:])
                for i in range(len(new_opt_par["param_groups"][0][ 'params'])):
                    index = new_opt_par["param_groups"][0]['params'][i] #第i层参数的索引
                    index2 = par["param_groups"][0]['params'][i] #第i层参数的索引
                    new_opt_par["state"][index]['step'] += par["state"][index2]['step'] * (p_i / self.C)
                    new_opt_par["state"][index]['exp_avg'] += par["state"][index2]['exp_avg']* (p_i / self.C)
                    new_opt_par["state"][index]['exp_avg_sq'] += par["state"][index2]['exp_avg_sq']* (p_i / self.C)

            #print(new_opt_par["state"]) #这是每层神经网络的参数:step,vt,mt,是一个Dict
            self.global_optimizer.load_state_dict(copy.deepcopy(new_opt_par))
            return self.global_model.state_dict().copy(),self.global_optimizer.state_dict().copy()
        return self.global_model.state_dict().copy(),self.global_model.state_dict().copy()


    def broadcast(self, new_par,new_opt_par): 
        """Send aggregated model to all clients"""
        for client in self.clients:
            client.recv(new_par.copy(),new_opt_par.copy())

    async def global_update(self):
        idxs_users = np.random.choice(range(len(self.clients)), int(self.C * len(self.clients)), replace=False)
        import asyncio

        results = await asyncio.gather(
            *[
                self.clients[idx].update() for idx in idxs_users
            ]
        )
        '''
        for idx in idxs_users:
            self.clients[idx].update()
        '''
        new_model,new_opt = self.aggregated(idxs_users)
        self.broadcast(new_model,new_opt)
        # acc = self.test_acc()
        acc = self.test_acc_femnist()
        return acc


    def test_acc_femnist(self):
        self.global_model.eval()
        correct = 0
        tot_sample = 0
        for i in range(len(self.data)):
            t_pred_y = self.global_model(self.data[i])
            _, predicted = torch.max(t_pred_y, 1)
            predicted= predicted
            correct += (predicted == self.target[i]).sum().item()
            tot_sample += self.target[i].size(0)
        acc = correct / tot_sample
        return acc

    def set_lr(self, lr):
        for c in self.clients:
            c.lr = lr
