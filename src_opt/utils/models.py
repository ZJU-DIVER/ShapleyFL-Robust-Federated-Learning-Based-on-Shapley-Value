#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

from torch import nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(MLP, self).__init__()
        self.layer_input = nn.Linear(dim_in, dim_hidden)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.layer_hidden = nn.Linear(dim_hidden, dim_out)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.view(-1, x.shape[1] * x.shape[-2] * x.shape[-1])
        x = self.layer_input(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.layer_hidden(x)
        return self.softmax(x)


class CNNMnist(nn.Module):
    def __init__(self, args):
        super(CNNMnist, self).__init__()
        self.conv1 = nn.Conv2d(args.num_channels, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, args.num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3])
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class CNNFashion_Mnist(nn.Module):
    def __init__(self, args):
        super(CNNFashion_Mnist, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.fc = nn.Linear(7 * 7 * 32, 10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return F.log_softmax(out, dim=1)


# class CNNCifar(nn.Module):
#     def __init__(self, args):
#         super(CNNCifar, self).__init__()
#         self.conv1 = nn.Conv2d(3, 6, 5)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(6, 16, 5)
#         self.fc1 = nn.Linear(16 * 5 * 5, 120)
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, args.num_classes)
#
#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = x.view(-1, 16 * 5 * 5)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return F.log_softmax(x, dim=1)


class CNNCifar(nn.Module):
    def __init__(self, args):
        super(CNNCifar,self).__init__()
        '''输入为3*32*32，尺寸减半是因为池化层'''
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)   #输出为16*16*16
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)  #输出为32*8*8
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 10)
        self.dropout = nn.Dropout(0.2)     #防止过拟合
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 8 * 8)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


'''
class CNNCifar(nn.Module):

    def __init__(self, conv1_dim=100, conv2_dim=150, conv3_dim=250, conv4_dim=500, *args, **kwargs):
        super(CNNCifar, self).__init__()

        self.conv4_dim = conv4_dim

        self.conv1 = nn.Conv2d(3, conv1_dim, 5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(conv1_dim, conv2_dim, 3, stride=1, padding=2)
        self.conv3 = nn.Conv2d(conv2_dim, conv3_dim, 3, stride=1, padding=2)
        self.conv4 = nn.Conv2d(conv3_dim, conv4_dim, 3, stride=1, padding=2)

        self.pool = nn.MaxPool2d(2, 2)
        # 3x3 is precalculated and written, you need to do it if you want to change the of filters
        self.fc1 = nn.Linear(conv4_dim * 3 * 3, 270)
        self.fc2 = nn.Linear(270, 150)
        self.fc3 = nn.Linear(150, 10)

        self.normalize1 = mbn.MyBatchNorm2d(conv1_dim)
        self.normalize2 = mbn.MyBatchNorm2d(conv2_dim)
        self.normalize3 = mbn.MyBatchNorm2d(conv3_dim)
        self.normalize4 = mbn.MyBatchNorm2d(conv4_dim)

    def forward(self, x, *args, **kwargs):
        # alpha = 1
        if len(args) != 0:
            # print("-------------------I AM HERE--------------", args[0][0])
            self.normalize1.running_mean = args[0][0]  # * alpha
            self.normalize2.running_mean = args[0][1]  # * alpha
            self.normalize3.running_mean = args[0][2]  # * alpha
            self.normalize4.running_mean = args[0][3]  # * alpha

            self.normalize1.running_var = args[1][0]  # * alpha
            self.normalize2.running_var = args[1][1]  # * alpha
            self.normalize3.running_var = args[1][2]  # * alpha
            self.normalize4.running_var = args[1][3]  # * alpha

        x, rm1, rv1 = self.normalize1((self.conv1(x)))
        x = self.pool(F.relu(x))  # first convolutional then batch normalization then relu then max pool
        x, rm2, rv2 = self.normalize2((self.conv2(x)))
        x = self.pool(F.relu(x))
        x, rm3, rv3 = self.normalize3((self.conv3(x)))
        x = self.pool(F.relu(x))
        x, rm4, rv4 = self.normalize4((self.conv4(x)))
        x = self.pool(F.relu(x))

        x = x.view(-1, self.conv4_dim * 3 * 3)  # flattening the features
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        
        print("==> printing bn1 running mean from SELF. during forward")
        print(self.normalize1.running_mean)
        
        return x, [rm1, rm2, rm3, rm4], [rv1, rv2, rv3, rv4]


class modelC(nn.Module):
    def __init__(self, input_size, n_classes=10, **kwargs):
        super(AllConvNet, self).__init__()
        self.conv1 = nn.Conv2d(input_size, 96, 3, padding=1)
        self.conv2 = nn.Conv2d(96, 96, 3, padding=1)
        self.conv3 = nn.Conv2d(96, 96, 3, padding=1, stride=2)
        self.conv4 = nn.Conv2d(96, 192, 3, padding=1)
        self.conv5 = nn.Conv2d(192, 192, 3, padding=1)
        self.conv6 = nn.Conv2d(192, 192, 3, padding=1, stride=2)
        self.conv7 = nn.Conv2d(192, 192, 3, padding=1)
        self.conv8 = nn.Conv2d(192, 192, 1)

        self.class_conv = nn.Conv2d(192, n_classes, 1)

    def forward(self, x):
        x_drop = F.dropout(x, .2)
        conv1_out = F.relu(self.conv1(x_drop))
        conv2_out = F.relu(self.conv2(conv1_out))
        conv3_out = F.relu(self.conv3(conv2_out))
        conv3_out_drop = F.dropout(conv3_out, .5)
        conv4_out = F.relu(self.conv4(conv3_out_drop))
        conv5_out = F.relu(self.conv5(conv4_out))
        conv6_out = F.relu(self.conv6(conv5_out))
        conv6_out_drop = F.dropout(conv6_out, .5)
        conv7_out = F.relu(self.conv7(conv6_out_drop))
        conv8_out = F.relu(self.conv8(conv7_out))

        class_out = F.relu(self.class_conv(conv8_out))
        pool_out = F.adaptive_avg_pool2d(class_out, 1)
        pool_out.squeeze_(-1)
        pool_out.squeeze_(-1)
        return pool_out
'''


class RnnLm(nn.Module):
    def __init__(self, args):
        super(RnnLm, self).__init__()
        self.args = args
        if not args.tied:
            self.embed = nn.Embedding(args.nvocab, args.d_embed)
        self.encoder = nn.GRU(args.d_embed, args.rnn_hidden, args.rnn_layers,
                              dropout=args.rnn_dropout, bias=True, bidirectional=False)
        self.fc1 = nn.Linear(args.rnn_hidden, args.nvocab, bias=True)

    def get_embedded(self, word_indexes):
        if self.args.tied:
            return self.fc1.weight.index_select(0, word_indexes)
        else:
            return self.embed(word_indexes)

    def forward(self, packed_sents):
        embedded_sents = nn.utils.rnn.PackedSequence(self.get_embedded(packed_sents.data), packed_sents.batch_sizes)
        out_packed_sequence, _ = self.encoder(embedded_sents)
        out = self.fc1(out_packed_sequence.data)
        return F.log_softmax(out, dim=1)
