# Federated Learning Model in PyTorch
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from DPMechanisms import gaussian_noise_weight,gaussian_noise, gaussian_noise_ls, clip_grad

import numpy as np
import copy
import random


class FLClient(nn.Module):
    """ Client of Federated Learning framework.
        1. Receive global model from server
        2. Perform local training (compute gradients)
        3. Return local model (gradients) to server
    """
    def __init__(self, model, output_size, data, lr, E, batch_size, clip, epsilon,delta, device=None):
    #def __init__(self, model, output_size, data, lr, E, batch_size, clip, sigma, device=None):
        """
        :param model: ML model's training process should be implemented
        :param data: (tuple) dataset, all data in client side is used as training data
        :param lr: learning rate
        :param E: epoch of local update
        """
        super(FLClient, self).__init__()
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
        self.noise = None
        self.lr = lr
        self.E = E
        self.clip = clip
        #self.sigma = sigma
        self.epsilon = epsilon
        self.delta = delta
        print()
        print("data[0].shape[1]",data[0].shape[1])
        self.model = model(data[0].shape[1], output_size).to(self.device)
        self.batch_model = model(data[0].shape[1], output_size).to(self.device)
        self.recv_model = model(data[0].shape[1], output_size).to(self.device)

    def recv(self, model_par):
        """receive global model from aggregator (server)"""
        self.model.load_state_dict(copy.deepcopy(model_par))
        self.recv_model.load_state_dict(copy.deepcopy(model_par))

    def update(self):
        """local model update"""
        self.model.train()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.lr)

        #optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9)
        for e in range(self.E):
            for batch_x, batch_y in self.data_loader:
                #print("batch_x",batch_x)
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                pred_y = self.model(batch_x)
                #print("pred_y",pred_y)
                loss = criterion(pred_y, batch_y.long()) / len(self.data_loader)
                loss.backward()
            # bound l2 sensitivity (gradient clipping)
            grads = dict(self.model.named_parameters())
            for name in grads:
                grads[name].grad = clip_grad(grads[name].grad, self.clip)
            optimizer.step()
            optimizer.zero_grad()

        # Add Gaussian noise
        # 1. compute l2-sensitivity by Client Based DP-FedAVG Alg.
        # 2. add noise
        sensitivity = 2 * self.lr * self.clip / self.data_size + (self.E - 1) * 2 * self.lr * self.clip
        new_param = copy.deepcopy(self.model.state_dict())
        for name in new_param:
            new_param[name] = torch.zeros(new_param[name].shape).to(self.device)
            new_param[name] += 1.0 * self.model.state_dict()[name]
            new_param[name] += gaussian_noise_weight(self.model.state_dict()[name].shape, sensitivity,self.epsilon, self.delta, device=self.device)
        self.model.load_state_dict(copy.deepcopy(new_param))

    def update_grad(self):
        """local model update, return gradients"""
        self.model.train()
        grad = {}
        params = dict(self.model.named_parameters())
        for name in params:
            grad[name] = torch.zeros(params[name].shape).to(self.device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0)
        losses = []
        for e in range(self.E):
            for batch_x, batch_y in self.data_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                pred_y = self.model(batch_x)
                loss = criterion(pred_y, batch_y.long())

                optimizer.zero_grad()
                loss.backward()
                losses += [loss.item()]
                optimizer.step()

                for name in params:
                    grad[name] += copy.deepcopy(params[name].grad)
            losses = []
        return grad.copy()
        

class FLServer(nn.Module):
    """ Server of Federated Learning
        1. Receive model (or gradients) from clients
        2. Aggregate local models (or gradients)
        3. Compute global model, broadcast global model to clients
    """
    def __init__(self, fl_par):
        super(FLServer, self).__init__()

        self.device = fl_par['device']
        self.client_num = fl_par['client_num']
        self.C = fl_par['C']  # (float) C in [0, 1]
        self.clip = fl_par['clip']

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
        self.clients = [FLClient(fl_par['model'],
                                 fl_par['output_size'],
                                 fl_par['data'][i],
                                 fl_par['lr'],
                                 fl_par['E'],
                                 fl_par['batch_size'],
                                 fl_par['clip'],
                                 #fl_par['sigma'],
                                 fl_par['epsilon'],
                                 fl_par['delta'],
                                 self.device)
                        for i in range(self.client_num)]
        self.global_model = fl_par['model'](self.input_size, fl_par['output_size']).to(self.device)
        self.weight = np.array([client.data_size * 1.0 for client in self.clients])
        self.broadcast(self.global_model.state_dict())

    def aggregated(self, idxs_users):
        """FedAvg"""
        model_par = [self.clients[idx].model.state_dict() for idx in idxs_users]
        new_par = copy.deepcopy(model_par[0])
        for name in new_par:
            new_par[name] = torch.zeros(new_par[name].shape).to(self.device)
        for idx, par in enumerate(model_par):
            w = self.weight[idxs_users[idx]] / np.sum(self.weight[:])
            for name in new_par:
                # new_par[name] += par[name] * (self.weight[idxs_users[idx]] / np.sum(self.weight[idxs_users]))
                new_par[name] += par[name] * (w / self.C)
        self.global_model.load_state_dict(copy.deepcopy(new_par))
        return self.global_model.state_dict().copy()

    def broadcast(self, new_par):
        """Send aggregated model to all clients"""
        for client in self.clients:
            client.recv(new_par.copy())

    def test_acc(self):
        # compute accuracy using test set
        self.global_model.eval()
        t_pred_y = self.global_model(self.data)
        _, predicted = torch.max(t_pred_y, 1)

        acc = (predicted == self.target).sum().item() / self.target.size(0)
        return acc

    def test_acc_femnist(self):
        self.global_model.eval()
        correct = 0
        tot_sample = 0
        for i in range(len(self.data)):
            t_pred_y = self.global_model(self.data[i])
            _, predicted = torch.max(t_pred_y, 1)
            #print("predicted",type(predicted))
            #print("predicted",predicted.shape)
            predicted= predicted
            print("predicted",predicted)

            print("self.target[i]",self.target[i])
            correct += (predicted == self.target[i]).sum().item()
            tot_sample += self.target[i].size(0)
        acc = correct / tot_sample
        return acc

    def global_update(self):
        idxs_users = np.random.choice(range(len(self.clients)), int(self.C * len(self.clients)), replace=False)
        for idx in idxs_users:
            self.clients[idx].update()
        self.broadcast(self.aggregated(idxs_users))
        # acc = self.test_acc()
        acc = self.test_acc_femnist()
        return acc


    def aggregated_grad(self, idxs_users, grads):
        """FedAvg - Update model using gradients"""
        agg_grad = copy.deepcopy(grads[0])
        for name in agg_grad:
            agg_grad[name] = torch.zeros(agg_grad[name].shape).to(self.device)

        for idx, grad in enumerate(grads):
            w = self.weight[idxs_users[idx]] / np.sum(self.weight[idxs_users])
            for name in grad:
                g = gaussian_noise(grad[name], self.clip, self.epsilon, self.delta, device=self.device) * self.lr
                agg_grad[name] += g * (w / self.C)

        for name in self.global_model.state_dict():
            self.global_model.state_dict()[name] -= agg_grad[name]
        return self.global_model.state_dict().copy()

    def global_update_grad(self):
        for e in range(self.E):
            idxs_users = np.random.choice(range(len(self.clients)), int(self.C * len(self.clients)), replace=False)
            grads = []
            for idx in idxs_users:
                grads.append(copy.deepcopy(self.clients[idx].update_grad()))
            self.broadcast(self.aggregated_grad(idxs_users, grads))
            acc = self.test_acc()
            print("global epochs = {:d}, acc = {:.4f}".format(e + 1, acc))

    def set_lr(self, lr):
        for c in self.clients:
            c.lr = lr
