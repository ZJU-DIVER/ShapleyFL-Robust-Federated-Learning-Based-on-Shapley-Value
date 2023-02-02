import random
import numpy as np
import math

"""
    arms: The number of arms
    T: The number of selection rounds
"""
def rewards_stationary(arms,T):
    rewards = []
    for i in range(arms):
        rewards.append(np.random.binomial(1,i/arms,T))
    return rewards

"""
    arms: The number of arms
    T: The number of selection rounds
"""
def rewards_non_stationary(arms,T):
    rewards = []
    for i in range(arms):
        expection = i / arms
        a = np.random.binomial(1, expection, int(T/2))
        if i >= arms/2:
            expection -= 0.2
        else:
            expection += 0.2
        b = np.random.binomial(1, expection, int(T/2))
        c = np.concatenate((a,b))
        rewards.append(c)
    return rewards

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

def simulation(arms,C,T,type):
    rewards = []
    if type == 'stationary':
        rewards = rewards_stationary(arms,T)
    else:
        rewards = rewards_non_stationary(arms,T)
    gamma = np.sqrt(2*math.log(arms)/arms/T)/2
    eta = np.sqrt(2*math.log(arms)/arms/T)
    S = np.ones(arms)
    P = np.ones(arms)
    for i in range(T):
        denominator = 0
        for j in range(arms):
            denominator += np.exp(eta*S[i])
        for j in range(arms):
            P[j] = np.exp(eta*S[i])/denominator
        selected_arms = arms_selection(P,C)
        for j in selected_arms:
            biased_l = (1-rewards[i][j])/(Q+gamma)
            S[i][j] += 1-biased_l













