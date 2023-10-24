from tqdm import trange, tqdm
from src_opt.utils.update import test_inference
from src_opt.utils.tools import average_weights
import numpy as np
import random
import heapq
import time

class Shapley():
    def __init__(self,local_weights,args, global_model, valid_dataset, init_acc):
        self.local_weights = local_weights
        self.args = args
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
        left_acc, current_loss = test_inference(self.args, self.global_model, self.valid_dataset)
        right_weights = self.get_weights_right(left_length, index, self.local_weights)
        if len(right_weights) > 0:
            right_weight = average_weights(right_weights)
            self.global_model.load_state_dict(right_weight)
            self.global_model.eval()
            right_acc, current_loss = test_inference(self.args, self.global_model, self.valid_dataset)
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
            current_acc, current_loss = test_inference(self.args, self.global_model, self.valid_dataset)
            for i in idx:
                shapley[i] += coef[len(idx)-1]*current_acc
            for i in set(coalition)-set(idx):
                shapley[i] -= coef[len(idx)]*current_acc
            enum(l)

        for i in range(len(shapley)):
            shapley[i] -= self.init_acc/len(self.local_weights)

        return shapley

    """
        Approximate Shapley value by monte carlo method
    """
    def eval_mcshap(self,subnumber):
        shapley = np.zeros(len(self.local_weights))
        for step in trange(subnumber):
            index = np.random.permutation(len(self.local_weights))
            original_acc = self.init_acc
            for j in range(1, len(index)+1):
                test_weights = self.get_weights(j, index, self.local_weights)
                test_weight = average_weights(test_weights)
                self.global_model.load_state_dict(test_weight)
                self.global_model.eval()
                current_acc, current_loss = test_inference(self.args, self.global_model, self.valid_dataset)
                shapley[index[j - 1]] += current_acc - original_acc
                original_acc = current_acc

        shapley = [shap / subnumber for shap in shapley]
        return shapley

    def eval_mcshap_time(self,time_budget):
        shapley = np.zeros(len(self.local_weights))
        time_start = time.time()
        Max = int(1e9)
        cnt = 0
        for step in trange(Max):
            time_end = time.time()
            if time_end - time_start > time_budget:
                break
            cnt += 1
            index = np.random.permutation(len(self.local_weights))
            original_acc = self.init_acc
            for j in range(1, len(index)+1):
                test_weights = self.get_weights(j, index, self.local_weights)
                test_weight = average_weights(test_weights)
                self.global_model.load_state_dict(test_weight)
                self.global_model.eval()
                current_acc, current_loss = test_inference(self.args, self.global_model, self.valid_dataset)
                shapley[index[j - 1]] += current_acc - original_acc
                original_acc = current_acc

        shapley = [shap / cnt for shap in shapley]
        return shapley

    """
        Approximate Shapley value by neyman method
    """
    def eval_neymanshap(self,subnumber):
        SV_estimator = np.zeros([len(self.local_weights)+1,len(self.local_weights)+1])
        cnt = np.zeros([len(self.local_weights)+1,len(self.local_weights)+1])
        marginal_contributions = [[] for i in range(len(self.local_weights)+1)]
        sampling_variance = np.zeros(len(self.local_weights)+1)
        sampling_number = np.zeros(len(self.local_weights)+1)

        for step in trange(int(subnumber/2)):
            index = np.random.permutation(len(self.local_weights))
            original_acc = self.init_acc
            for j in range(1, len(index)+1):
                test_weights = self.get_weights(j, index, self.local_weights)
                test_weight = average_weights(test_weights)
                self.global_model.load_state_dict(test_weight)
                self.global_model.eval()
                current_acc, current_loss = test_inference(self.args, self.global_model, self.valid_dataset)
                SV_estimator[index[j - 1]][j] += current_acc - original_acc
                cnt[index[j - 1]][j] += 1
                marginal_contributions[j].append(current_acc - original_acc)
                original_acc = current_acc

        for i in range(len(marginal_contributions)):
            if len(marginal_contributions[i]) != 0:
                mu = np.array(marginal_contributions[i]).sum()
                for j in range(len(marginal_contributions[i])):
                    sampling_variance[i] += (marginal_contributions[i][j]-mu)**2/(len(marginal_contributions)-1)

        for j in trange(1, len(sampling_variance)):
            sampling_number[j] = int(sampling_variance[j] / np.array(sampling_variance).sum() * subnumber / 2 * len(self.local_weights))
            print(sampling_number[j])

        for j in trange(1, len(sampling_variance)):
            sampling_number[j] = int(sampling_variance[j] / np.array(sampling_variance).sum() * subnumber/2*len(self.local_weights))

            for _ in range(int(sampling_number[j])):
                index = np.random.permutation(len(self.local_weights))
                original_acc = self.init_acc
                if j > 1:
                    test_weights = self.get_weights(j-1, index, self.local_weights)
                    test_weight = average_weights(test_weights)
                    self.global_model.load_state_dict(test_weight)
                    self.global_model.eval()
                    original_acc, original_loss = test_inference(self.args, self.global_model, self.valid_dataset)

                test_weights = self.get_weights(j, index, self.local_weights)
                test_weight = average_weights(test_weights)
                self.global_model.load_state_dict(test_weight)
                self.global_model.eval()
                current_acc, current_loss = test_inference(self.args, self.global_model, self.valid_dataset)
                SV_estimator[index[j - 1]][j] += current_acc - original_acc
                cnt[index[j - 1]][j] += 1

        shapley = np.zeros(len(self.local_weights))
        for i in range(len(self.local_weights)):
            for j in range(len(self.local_weights)+1):
                if cnt[i][j] != 0:
                    shapley[i] += SV_estimator[i][j] / cnt[i][j] / len(self.local_weights)

        return shapley
    def eval_ccshap(self, subnumber):
        shapley = np.zeros(len(self.local_weights))
        length = len(self.local_weights)
        for step in trange(int(subnumber * length)):
            index = np.random.permutation(len(self.local_weights))
            j = random.randint(1, len(shapley))
            left_weights = self.get_weights(j, index, self.local_weights)
            left_weight = average_weights(left_weights)
            self.global_model.load_state_dict(left_weight)
            self.global_model.eval()
            left_acc, current_loss = test_inference(self.args, self.global_model, self.valid_dataset)
            right_weights = self.get_weights_right(j, index, self.local_weights)
            if len(right_weights) > 0:
                right_weight = average_weights(right_weights)
                self.global_model.load_state_dict(right_weight)
                self.global_model.eval()
                right_acc, current_loss = test_inference(self.args, self.global_model, self.valid_dataset)
            else:
                right_acc = self.init_acc
            for k in range(len(index)):
                if k < j:
                    if j == len(shapley):
                        shapley[index[k]] += (left_acc - right_acc) / len(shapley)
                    else:
                        shapley[index[k]] += (left_acc - right_acc) / (2 * j)
                else:
                    shapley[index[k]] += (right_acc - left_acc) / (2 * (len(index) - j))
        shapley = [shap / subnumber for shap in shapley]
        return shapley

    def eval_ccshap_time(self, time_budget):
        shapley = np.zeros(len(self.local_weights))
        length = len(self.local_weights)
        time_start = time.time()
        Max = int(1e9)
        cnt = 0
        for step in trange(Max):
            if cnt%length == 0:
                time_end = time.time()
                if time_end-time_start > time_budget:
                    break
            cnt += 1
            index = np.random.permutation(len(self.local_weights))
            j = random.randint(1, len(shapley))
            left_weights = self.get_weights(j, index, self.local_weights)
            left_weight = average_weights(left_weights)
            self.global_model.load_state_dict(left_weight)
            self.global_model.eval()
            left_acc, current_loss = test_inference(self.args, self.global_model, self.valid_dataset)
            right_weights = self.get_weights_right(j, index, self.local_weights)
            if len(right_weights) > 0:
                right_weight = average_weights(right_weights)
                self.global_model.load_state_dict(right_weight)
                self.global_model.eval()
                right_acc, current_loss = test_inference(self.args, self.global_model, self.valid_dataset)
            else:
                right_acc = self.init_acc
            for k in range(len(index)):
                if k < j:
                    if j == len(shapley):
                        shapley[index[k]] += (left_acc - right_acc) / len(shapley)
                    else:
                        shapley[index[k]] += (left_acc - right_acc) / (2 * j)
                else:
                    shapley[index[k]] += (right_acc - left_acc) / (2 * (len(index) - j))
        shapley = [shap / cnt * length for shap in shapley]
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







