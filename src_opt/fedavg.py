# Version 2.0
import sys, os
cur_path = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, cur_path+"/..")

import os
import copy
import time
import pickle
import numpy as np
from tqdm import tqdm

import torch
from tensorboardX import SummaryWriter
import torch.nn.functional as F

from src_opt.utils.options import args_parser
from src_opt.utils.update import LocalUpdate, test_inference
from src_opt.utils.models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar
from src_opt.utils.tools import get_dataset, average_weights, exp_details, add_gradient_noise, add_random_gradient, get_noiseword

args = args_parser()
exp_details(args)

def solver():
    start_time = time.time()

    # define paths
    path_project = os.path.abspath('')
    logger = SummaryWriter('../logs')

    # if args.gpu_id:
    #     torch.cuda.set_device(args.gpu_id)
    # device = 'cuda' if args.gpu else 'cpu'

    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != None else 'cpu')
    train_dataset, valid_dataset, test_dataset, user_groups = get_dataset(args)

    # BUILD MODEL
    if args.model == 'cnn':
        # Convolutional neural netork
        if args.dataset == 'mnist':
            global_model = CNNMnist(args=args)
        elif args.dataset == 'fmnist':
            global_model = CNNFashion_Mnist(args=args)
        elif args.dataset == 'cifar':
            global_model = CNNCifar(args=args)

    elif args.model == 'mlp':
        # Multi-layer perceptron
        img_size = train_dataset[0][0].shape
        len_in = 1
        for x in img_size:
            len_in *= x
            global_model = MLP(dim_in=len_in, dim_hidden=64,
                               dim_out=args.num_classes)
    else:
        exit('Error: unrecognized model')

    # Set the model to train and send it to device.
    global_model = global_model.to(args.device)
    global_model.train()

    # copy weights
    global_weights = global_model.state_dict()
    original_weights = copy.copy(global_weights)
    # Training
    train_loss, train_accuracy = [], []
    allAcc_list = []
    print_every = 2

    accuracy_list = []

    for epoch in tqdm(range(args.epochs)):
        local_weights, local_losses = [], []
        print(f'\n | Global Training Round : {epoch + 1} |\n')

        global_model.train()
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)

        for idx in idxs_users:
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[idx], logger=logger)
            w, loss = local_model.update_weights(
                model=copy.deepcopy(global_model).to(args.device), global_round=epoch)
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))

        # update global weights
        # local_weights.append(copy.deepcopy(global_weights))
        # Add Gradient Noise
        print(local_weights[0])
        local_weights = add_gradient_noise(args, local_weights, idxs_users)
        global_weights = average_weights(local_weights)
        # if epoch < 75:
        #     global_weights = average_weights(local_weights)
        # elif epoch < 100:
        #     shapley = np.ones(m)
        #     shapley = F.softmax(torch.tensor(shapley), dim=0)
        #     global_weights = SVAtt_weights(local_weights, shapley, original_weights, 0.1, epoch)

        # update global weights
        global_model.load_state_dict(global_weights)
        original_weights = copy.copy(global_weights)

        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)

        # Calculate avg training accuracy over all users at every epoch
        list_acc, list_loss = [], []
        global_model.eval()
        for c in range(args.num_users):
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[c], logger=logger)
            acc, loss = local_model.inference(model=global_model)
            list_acc.append(acc)
            list_loss.append(loss)
        train_accuracy.append(sum(list_acc) / len(list_acc))

        # print global training loss after every 'i' rounds
        if (epoch + 1) % print_every == 0:
            print(f' \nAvg Training Stats after {epoch + 1} global rounds:')
            print(f'Training Loss : {np.mean(np.array(train_loss))}')
            print('Train Accuracy: {:.2f}% \n'.format(100 * train_accuracy[-1]))
        test_acc, test_loss = test_inference(args, global_model, test_dataset)
        allAcc_list.append(test_acc)
        print(" \nglobal accuracy:{:.2f}%".format(100 * test_acc))

    #draw(args.epochs, allAcc_list, "FedAvg 10 100")
    # Test inference after completion of training
    test_acc, test_loss = test_inference(args, global_model, test_dataset)

    accuracy_list.append(test_acc)

    print(f' \n Results after {args.epochs} global rounds of training:')
    print("|---- Avg Train Accuracy: {:.2f}%".format(100 * train_accuracy[-1]))
    print("|---- Test Accuracy: {:.2f}%".format(100 * test_acc))

    # Saving the objects train_loss and train_accuracy:
    file_name = '../save/objects/{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}].pkl'. \
        format(args.dataset, args.model, args.epochs, args.frac, args.iid,
               args.local_ep, args.local_bs)

    with open(file_name, 'wb') as f:
        pickle.dump([train_loss, train_accuracy], f)

    print('\n Total Run Time: {0:0.4f}'.format(time.time() - start_time))
    return test_acc, train_accuracy[-1], allAcc_list

def show_avg(list):
    ans = []
    ans.append(np.mean(list[17:22]))
    ans.append(np.mean(list[37:42]))
    ans.append(np.mean(list[57:62]))
    ans.append(np.mean(list[77:82]))
    ans.append(np.mean(list[95:]))
    print(ans)

if __name__ == '__main__':
    test_acc, train_acc = 0, 0
    repeat = 5
    noise = args.noise
    NoiseWord = get_noiseword()
    for _ in range(repeat):
        print("|---- Repetition {} ----|".format(_ + 1))
        test, train, acc_list = solver()
        test_acc += test
        train_acc += train
        # show_avg(acc_list)
        path = '../save_opt/{}/FedAvg_{}_cnn_E{}_N{}_repeat{}_{}.txt'.format(NoiseWord[noise], args.dataset, args.epochs,  args.noiselevel, repeat, args.device)
        f = open(path, "a+")
        f.writelines("Repetition [%d] : [%s]\n" % (_ + 1, ', '.join(["%.4f" % w for w in acc_list])))
        f.flush()
        f.close()
    print('|---------------------------------')
    print("|---- Train Accuracy: {:.2f}%".format(100 * (train_acc / repeat)))
    print("|---- Test Accuracy: {:.2f}%".format(100 * (test_acc / repeat)))