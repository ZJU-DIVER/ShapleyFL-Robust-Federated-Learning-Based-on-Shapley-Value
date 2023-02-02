from tqdm import trange, tqdm
from MedicalDiagnosis.utils import evaluate_model_on_tests
import numpy as np
import random
import heapq
import time
import copy
import torch

from MedicalDiagnosis.datasets.fed_isic2019 import (
    metric,
)
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

class Shapley():
    def __init__(self,local_weights, global_model, valid_dataset, init_acc):
        self.local_weights = local_weights
        self.global_model = global_model
        self.valid_dataset = valid_dataset
        self.init_acc = init_acc

    def get_weights(self,j, idx, local_ws):
        test_weights = []
        for i in range(j):
            current_weight = local_ws[idx[i]]
            test_weights.append(current_weight)

        return test_weights

    def get_weights_right(self,j,idx,local_ws):
        test_weights = []
        for i in range(j,len(idx)):
            current_weight = local_ws[idx[i]]
            test_weights.append(current_weight)
        return test_weights

    def get_acc(self, index, left_length):
        if left_length == -1:
            left_length = random.randint(1, len(index))
        left_weights = self.get_weights(left_length, index, self.local_weights)
        left_weight = average_weights(left_weights)
        self.global_model.load_state_dict(left_weight)
        self.global_model.eval()
        left_acc = evaluate_model_on_tests(self.global_model, self.valid_dataset, metric)["client_test_0"]
        right_weights = self.get_weights_right(left_length, index, self.local_weights)
        if len(right_weights) > 0:
            right_weight = average_weights(right_weights)
            self.global_model.load_state_dict(right_weight)
            self.global_model.eval()
            right_acc = evaluate_model_on_tests(self.global_model, self.valid_dataset,metric)["client_test_0"]
        else:
            right_acc = self.init_acc

        return left_acc, right_acc

    """
        Calculate the exact Shapley value
    """
    def eval_exactshap(self):
        n = len(self.local_weights)

        def enum(l):
            for i in range(len(self.local_weights)-1,-1,-1):
                if l[i] == 0:
                    l[i] = 1
                    break
                else:
                    l[i] = 0
            return l

        shapley = np.zeros(n)
        coef = np.zeros(n)
        fact = np.math.factorial
        coalition = np.arange(n)
        for s in range(n):
            coef[s] = fact(s)*fact(n-s-1)/fact(n)
        l = np.zeros(n)

        enum(l)
        while np.sum(l) != 0:
            idx = []
            test_weights = []
            for i in range(n):
                if l[i] == 1:
                    idx.append(i)
                    test_weights.append(self.local_weights[i])
            test_weight = average_weights(test_weights)
            self.global_model.load_state_dict(test_weight)
            self.global_model.eval()
            current_acc = evaluate_model_on_tests(self.global_model, self.valid_dataset, metric)["client_test_0"]
            for i in idx:
                shapley[i] += coef[len(idx)-1]*current_acc
            for i in set(coalition)-set(idx):
                shapley[i] -= coef[len(idx)]*current_acc
            enum(l)

        for i in range(len(shapley)):
            shapley[i] -= self.init_acc/len(self.local_weights)

        return shapley
    def eval_ccshap_stratified(self,subnumber):
        length = len(self.local_weights)
        shapley = np.zeros(length)
        shapley_estimator = [[[] for i in range(length)] for j in range(length)]

        #init
        for i in trange(subnumber):
            for j in range(length):
                index = np.random.permutation(len(self.local_weights))
                left_acc, right_acc = self.get_acc(index,j+1)
                for k in range(len(index)):
                    if k <= j:
                        shapley_estimator[index[k]][j].append(left_acc-right_acc)
                    else:
                        shapley_estimator[index[k]][length-j-2].append(right_acc-left_acc)

        for i in range(length):
            for j in range(length):
                if len(shapley_estimator[i][j]) > 0:
                    shapley[i] += np.mean(shapley_estimator[i][j])/length

        return shapley
















