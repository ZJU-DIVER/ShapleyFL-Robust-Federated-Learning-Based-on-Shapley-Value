import copy
import torch
import ssl
import random
import numpy as np
import math

# def LabelNoise():
#     # label noise
#     # train_dataset.train_labels[train_dataset.train_labels == 0] = 1
#     train_dataset.targets = torch.tensor(train_dataset.targets)
#     for i in range(0, 9, 2):
#         train_dataset.targets[train_dataset.targets == i] = i + 1
#         indices = []
#     for i in range(len(valid_dataset)):
#         data, label = valid_dataset[i]
#         if label % 2 != 0:
#             indices.append(i)
#     new_valid_dataset = torch.utils.data.Subset(valid_dataset, indices)
#     valid_dataset = new_valid_dataset

#     indices = []
#     for i in range(len(test_dataset)):
#         data, label = test_dataset[i]
#         if label % 2 != 0:
#             indices.append(i)
#     new_test_dataset = torch.utils.data.Subset(test_dataset, indices)
#     test_dataset = new_test_dataset

def GradientNoise(w, noisetype, noiselevel, device, noiseseed):
    if (noiseseed == 21):
        noiseclient = [0,5]
    elif(noiseseed == 20):
        noiseclient = [0,1]
    else:
        random.seed(noiseseed)
        noiseclient = random.sample(range(0,6), 2)
    print(noiseclient)
    for i in range(len(w)):
        for key in w[i].keys():
            if i in noiseclient:
                if noisetype == 5:
                    noise = torch.tensor(np.random.normal(0, noiselevel, w[i][key].shape)).to(device)
                    ratio = torch.ones(w[i][key].shape).to(device)
                    w[i][key] = w[i][key] * (ratio + noise)
                if noisetype == 6:
                    noise = torch.tensor(np.random.normal(0, noiselevel, w[i][key].shape))
                    noise = noise.to(torch.float32)
                    noise = noise.to(device)
                    # print("original weight = ", w[i][key])
                    w[i][key] = noise
                if noisetype == 7:
                    w[i][key] = w[i][key] * -10
                if noisetype == 8:
                    w[i][key] = torch.ones(w[i][key].shape) * -1
    return w
