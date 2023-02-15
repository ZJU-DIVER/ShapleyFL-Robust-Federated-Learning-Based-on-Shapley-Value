#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import copy
import torch
from torchvision import datasets, transforms
from src_opt.utils.sampling import mnist_iid, mnist_noniid, mnist_noniid_unequal, cifar_iid, cifar_noniid, \
    FashionMnist_noniid, cifar_longtail
from src_opt.utils.options import args_parser
import ssl
import random
import numpy as np
import math

def get_noiseword():
    NoiseWord = ["0_NonIID", "1_LongTail", "2_LabelNoise", "3_LabelNoise2", "4_DataNoise", "5_GradientNoise", "6_RandomAttack", "7_ReverseGradient", "8_ConstantAttack"]
    return NoiseWord

def get_datasetserver(args):
    if args.dataset == 'cifar':
        data_dir = '../data/cifar/'
        apply_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        test_dataset_all = datasets.CIFAR10(data_dir, train=False, download=True,
                                            transform=apply_transform)
        train_dataset, test_dataset = torch.utils.data.random_split(test_dataset_all, [args.sz, 10000-args.sz])
        return train_dataset, test_dataset

    elif args.dataset == 'fmnist':
        data_dir = '../data/fmnist'
        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])
        test_dataset_all = datasets.FashionMNIST(data_dir, train=False, download=True,
                                            transform=apply_transform)
        train_dataset, test_dataset = torch.utils.data.random_split(test_dataset_all, [args.sz, 10000 - args.sz])
        return train_dataset, test_dataset


def get_dataset(args):
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """
    ssl._create_default_https_context = ssl._create_unverified_context

    if args.dataset == 'cifar':
        data_dir = '../data/cifar/'
        apply_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True,
                                         transform=apply_transform)

        test_dataset_all = datasets.CIFAR10(data_dir, train=False, download=True,
                                            transform=apply_transform)
        test_dataset, valid_dataset = torch.utils.data.random_split(test_dataset_all, [8000, 2000])

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = cifar_iid(train_dataset, args.num_users)
        else:
            # Long-tailed
            if args.noise == 1:
                user_groups = cifar_longtail(train_dataset,  args.num_users, args.noiselevel)
            elif args.unequal:
                # Chose uneuqal splits for every user
                raise NotImplementedError()
            else:
                # Chose equal splits for every user
                user_groups = cifar_noniid(train_dataset, args.num_users)

    elif args.dataset == 'mnist':
        data_dir = '../data/mnist/'
        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])

        train_dataset = datasets.MNIST(data_dir, train=True, download=True,
                                       transform=apply_transform)

        test_dataset_all = datasets.MNIST(data_dir, train=False, download=True,
                                          transform=apply_transform)

        test_dataset, valid_dataset = torch.utils.data.random_split(test_dataset_all, [8000, 2000])

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = mnist_iid(train_dataset, args.num_users)
        else:
            # Sample 0_NonIID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                user_groups = mnist_noniid_unequal(train_dataset, args.num_users)
            else:
                # Chose euqal splits for every user
                user_groups = mnist_noniid(train_dataset, args.num_users)

    elif args.dataset == 'fmnist':
        data_dir = '../data/fmnist'
        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])

        train_dataset = datasets.FashionMNIST(data_dir, train=True, download=True,
                                              transform=apply_transform)

        test_dataset_all = datasets.FashionMNIST(data_dir, train=False, download=True,
                                                 transform=apply_transform)

        test_dataset, valid_dataset = torch.utils.data.random_split(test_dataset_all, [8000, 2000])

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = mnist_iid(train_dataset, args.num_users)
        else:
            # Long-tailed
            if args.noise == 1:
                user_groups = cifar_longtail(train_dataset,  args.num_users, args.noiselevel)
            # Sample 0_NonIID user data from Mnist
            elif args.unequal:
                # Chose uneuqal splits for every user
                user_groups = mnist_noniid_unequal(train_dataset, args.num_users)
            else:
                # Chose euqal splits for every user
                user_groups = FashionMnist_noniid(train_dataset, args.num_users)

        # label noise
        # train_dataset.train_labels[train_dataset.train_labels == 0] = 1
        if args.noise == 2:
            train_dataset.targets = torch.tensor(train_dataset.targets)
            for i in range(0, 9, 2):
                train_dataset.targets[train_dataset.targets == i] = i + 1
                indices = []
            for i in range(len(valid_dataset)):
                data, label = valid_dataset[i]
                if label % 2 != 0:
                    indices.append(i)
            new_valid_dataset = torch.utils.data.Subset(valid_dataset, indices)
            valid_dataset = new_valid_dataset

            indices = []
            for i in range(len(test_dataset)):
                data, label = test_dataset[i]
                if label % 2 != 0:
                    indices.append(i)
            new_test_dataset = torch.utils.data.Subset(test_dataset, indices)
            test_dataset = new_test_dataset
        # label noise2
        elif args.noise == 3:
            new_train_dataset = []
            train_dataset.targets = torch.tensor(train_dataset.targets)
            for i in range(len(train_dataset)):
                feature, label = train_dataset[i]
                if int(i / 200) % 4 == 0:
                    noiselabel = (label + 1) % 10
                    new_train_dataset.append((feature, noiselabel))
                else:
                    new_train_dataset.append((feature, label))
            train_dataset = new_train_dataset
        # data noise
        elif args.noise == 4:
            new_train_dataset = []
            for i in range(len(train_dataset)):
                feature, label = train_dataset[i]
                if int(i / 200) % 2 == 0:
                    noise = torch.tensor(np.random.normal(0, 1, feature.shape))
                    noise = noise.to(torch.float32)
                    new_data = feature + noise
                    clip_data = torch.clamp(new_data, -1, 1)
                    new_train_dataset.append((clip_data, label))
                else:
                    new_train_dataset.append((feature, label))
            train_dataset = new_train_dataset

    return train_dataset, valid_dataset, test_dataset, user_groups

def add_gradient_noise(args, w, idxs):
    if (args.noise < 5):
        return w
    for i in range(len(w)):
        for key in w[i].keys():
            if idxs[i] % 4 == 0:
                if args.noise == 5:
                    noise = torch.tensor(np.random.normal(0, args.noiselevel, w[i][key].shape)).to(args.device)
                    ratio = torch.ones(w[i][key].shape).to(args.device)
                    w[i][key] = w[i][key] * (ratio + noise)
                if args.noise == 6:
                    noise = torch.tensor(np.random.normal(0, args.noiselevel, w[i][key].shape))
                    noise = noise.to(torch.float32)
                    noise = noise.to(args.device)
                    # print("original weight = ", w[i][key])
                    w[i][key] = noise
                if args.noise == 7:
                    w[i][key] = w[i][key] * -10
                if args.noise == 8:
                    w[i][key] = torch.ones(w[i][key].shape) * -1
    return w

def add_random_gradient(args, w, idxs):
    for i in range(len(w)):
        for key in w[i].keys():
            if idxs[i] % 10 == 0:
                # print(idxs[i])
                noise = torch.tensor(np.random.normal(0, args.noiselevel, w[i][key].shape))
                noise = noise.to(torch.float32)
                noise = noise.to(args.device)
                # print("original weight = ", w[i][key])
                w[i][key] = noise
                # print("noise weight = ", w[i][key])
    return w

def average_weights(w):
    """
    最正常的平均
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] = w_avg[key] + w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg

def avgSV_weights(w, shapley,ori):
    """
        Shapley权值平均
        Returns the average of the weights.
    """
    w_avg = copy.deepcopy(ori)
    for key in w_avg.keys():
        for i in range(0, len(w)):
            w_avg[key] = w_avg[key] + (w[i][key]-ori[key]) * shapley[i]
    return w_avg

def avgSV_baseline(w, shapley, ori):
    """
        FedSV Shapley权值平均 beta=0.5
        Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        w_avg[key] = w_avg[key] * shapley[0]
        for i in range(1, len(w)):
            w_avg[key] = w_avg[key] + w[i][key] * shapley[i]
        w_avg[key] = torch.div(w_avg[key], 2) + torch.div(ori[key], 2)
    return w_avg


"""
    p: The probabilities of each arm been picked in the first round
    C: The number of arms that been picked in each round
"""
def arms_selection(p,C):
    selected = []
    tuples = []
    for i in range(len(p)):
        tuples.append((i,p[i]))
    remain = 1
    for _ in range(C):
        rand = random.random()
        pre = 0
        for i in range(len(tuples)):
            if tuples[i][0] not in selected:
                if rand >= pre and rand < pre+tuples[i][1]/remain:
                    selected.append(i)
                    remain -= tuples[i][1]
                    break
                else:
                    pre += tuples[i][1]/remain
    return selected

def unbiased_selection(p):
    idxs = []
    while(len(idxs) < 2):
        idxs = []
        for i in range(len(p)):
            rand = random.random()
            if rand < p[i]:
                idxs.append(i)
    return idxs

def softmax(a,eta):
    s = 0
    p = np.zeros(len(a))
    for i in range(len(a)):
        s += math.exp(eta*a[i])
    for i in range(len(a)):
        p[i] = math.exp(eta*a[i])/s
    return p



def exp_details(args):
    print('\nExperimental details:')
    if args.gpu:
        print(f'    Environment   : CUDA {args.gpu}')
    else:
        print(f'    Environment   : CPU')
    print(f'    Dataset   : {args.dataset}')
    print(f'    Model     : {args.model}')
    print(f'    Optimizer : {args.optimizer}')
    print(f'    Learning  : {args.lr}')
    print(f'    Global Rounds   : {args.epochs}\n')

    print('    Federated parameters:')
    print(f'    Fraction of users  : {args.frac}')
    print(f'    Local Batch size   : {args.local_bs}')
    print(f'    Local Epochs       : {args.local_ep}\n')

    NoiseWord = get_noiseword()
    print('    Noise parameters:')
    if args.noise:
        print(f'    Noise  : {NoiseWord[args.noise]}')
        print(f'    NoiseLevel   : {args.noiselevel}')

    return
