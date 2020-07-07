#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
from sklearn import metrics


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


class LocalUpdate(object):
    def __init__(self, args, dataset=None, idxs=None, net=None, epochs=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)
        
        '''
        if epochs <= 200 : step_lr=0.1
        elif epochs <= 300 : step_lr=0.1
        else: step_lr=0.01
        '''
        
        #if epochs <= 250 : step_lr=0.01
        #else : step_lr=0.01
       
        

        self.optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=0.9, weight_decay=args.wd)
        self.net = net

        #optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=0.9, weight_decay=self.args.wd)
        #optimizer = torch.optim.SGD(net.parameters(), lr=0.016, momentum=0.9, weight_decay=5e-4)
        #optimizer = torch.optim.Adam(net.parameters(), lr=self.args.lr, weight_decay=0)
        #optimizer = torch.optim.Adam(net.parameters(), lr=0.001, weight_decay=self.args.wd)
        
    def train(self):
        self.net.train()
        # train and update
        

        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                self.net.zero_grad()
                log_probs = self.net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                self.optimizer.step()
                if self.args.verbose and batch_idx % 1 == 0:
                    print('\rUpdate Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(images), len(self.ldr_train.dataset),
                               100. * batch_idx / len(self.ldr_train), loss.item()), end='')
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return self.net, sum(epoch_loss) / len(epoch_loss)

