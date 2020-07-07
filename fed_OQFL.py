#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms
import torch
import os

from utils.sampling import mnist_iid, mnist_noniid, cifar_iid
from utils.options import args_parser
from models.Update import LocalUpdate
from models.Nets import MLP, CNNMnist, CNNCifar
from models.Fed import FedAvg
from models.test import test_img, file_store

#from models.mobilenetv2 import *

from model.vgg_quant_LSQ import *
from models.Master_vgg_LSQ import *
from models.mobilenetv2LSQ import *
from models.preact_resnet import *
from models.preact_resnetLSQ import *


def cifar_transform(is_training=True):
    if is_training:
        transform_list = [transforms.RandomHorizontalFlip(),
                      transforms.Pad(padding=4, padding_mode='reflect'),
                      transforms.RandomCrop(32, padding=0),
                      transforms.ToTensor(),
                      transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), ]
    else:
        transform_list = [transforms.ToTensor(),
                      transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), ]

    transform_list = transforms.Compose(transform_list)
    return transform_list


if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    args.iid = 1
    
    

    # load dataset and split users
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)
        # sample users
        if args.iid:
            dict_users = mnist_iid(dataset_train, args.num_users)
        else:
            dict_users = mnist_noniid(dataset_train, args.num_users)
    elif args.dataset == 'cifar':
        trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        dataset_train = datasets.CIFAR10('./data/cifar', train=True, transform=cifar_transform(is_training=True), download=True)
        dataset_test = datasets.CIFAR10('./data/cifar', train=False, transform=cifar_transform(is_training=False), download=True)
        #dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=trans_cifar)
        #dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=trans_cifar)
        if args.iid:
            dict_users = cifar_iid(dataset_train, args.num_users)
        else:
            exit('Error: only consider IID setting in CIFAR10')
    else:
        exit('Error: unrecognized dataset')
    img_size = dataset_train[0][0].shape

    # build model
    if args.model == 'cnn' and args.dataset == 'cifar':
        net_glob = CNNCifar(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'mnist':
        net_glob = CNNMnist(args=args).to(args.device)
    elif args.model == 'mlp':
        len_in = 1
        for x in img_size:
            len_in *= x
        net_glob = MLP(dim_in=len_in, dim_hidden=200, dim_out=args.num_classes).to(args.device)
    else:
        exit('Error: unrecognized model')
    #net_glob = ResNet18()

    net_best = 0
    net_glob = VGG('VGG16')


    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('ckpt'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./ckpt/vgg_full.pth')
        net_glob.load_state_dict(checkpoint['net'])
        net_best = checkpoint['acc']
        start_iter = checkpoint['iter']

        print('net_best: ',net_best)
        print('start_iter: ',start_iter)

    print(net_glob)

    nets_users = []
    for i in range(args.num_users):
        nets_users.append([0, copy.deepcopy(net_glob.state_dict())])
    

    # copy weights
    w_glob = net_glob.state_dict()

    # training
    loss_train = []
    cv_loss, cv_acc = [], []
    val_loss_pre, counter = 0, 0
    
    best_loss = None
    val_acc_list, net_list = [], []

    acc_train = []
    loss_tr = []
    acc_test = []
    loss_test = []

    for iter in range(args.epochs):
        #net_glob.train()
        w_locals, loss_locals = [], []
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        for order, idx in enumerate(idxs_users):
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx], net=copy.deepcopy(net_glob).to(args.device), epochs=iter)
            #w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))
            #w_locals.append(copy.deepcopy(w))
            #loss_locals.append(copy.deepcopy(loss))
            #w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))
            w, loss = local.train()
    
            print('\rEpochs: {}\tUserID: {}\tSequence: {}\tLoss: {:.6f}'.format(
                        iter, idx, order, loss))
            loss_locals.append(copy.deepcopy(loss))
            #w_locals.append(copy.deepcopy(w.state_dict()))
            nets_users[idx][0] = 1
            nets_users[idx][1] = copy.deepcopy(w.to(torch.device('cpu')).state_dict())
            
        # update global weights
        w_glob = FedAvg(nets_users)

        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)

        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
        loss_train.append(loss_avg)

        #print accuracy
        print("Apochs: ", iter)
        atr, ltr = test_img(net_glob.to(args.device), dataset_train, args)
        ate, lte = test_img(net_glob.to(args.device), dataset_test, args)
        
        if net_best < ate:
            print('Saving...')
            state={
                'net': net_glob.state_dict(),
                'acc': ate,
                'iter': iter
            }
            if not os.path.isdir('ckpt'):
                os.mkdir('ckpt')
            torch.save(state, './ckpt/vgg_full.pth')
            net_best = ate
            
        acc_train.append(atr)
        loss_tr.append(ltr)
        acc_test.append(ate)
        loss_test.append(lte)

    
    file_store(acc_train, loss_tr, acc_test, loss_test)

    # plot loss curve
    plt.figure()
    plt.plot(range(len(loss_train)), loss_train)
    plt.ylabel('train_loss')
    plt.savefig('./save/fed_{}_{}_{}_C{}_iid{}.png'.format(args.dataset, args.model, args.epochs, args.frac, args.iid))

    # testing
    net_glob.eval()
    acc_train, loss_train = test_img(net_glob, dataset_train, args)
    acc_test, loss_test = test_img(net_glob, dataset_test, args)
    print("Training accuracy: {:.2f}".format(acc_train))
    print("Testing accuracy: {:.2f}".format(acc_test))

