import argparse
import os
import scipy.io
import torch
import sys
import torch.nn.functional as F
import copy
import math
from torch.utils.tensorboard import SummaryWriter

import numpy as np
from scipy.stats import entropy
from scipy.spatial.distance import jensenshannon
import utils
import models



def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(20)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ASJDA():
    def __init__(self, model=models.ASJDANet(), source_loaders=0, target_loader=0, batch_size=256, iteration=10000, lr=0.01, momentum=0.9, log_interval=10):
        self.model = model
        self.model.to(device)
        self.source_loaders = source_loaders
        self.target_loader = target_loader
        self.batch_size = batch_size
        self.iteration = iteration
        self.lr = lr
        self.momentum = momentum
        self.log_interval = log_interval


    def __getModel__(self):
        return self.model

    def train(self):
        # best_model_wts = copy.deepcopy(model.state_dict())
        source_iters = []
        for i in range(len(self.source_loaders)):
            source_iters.append(iter(self.source_loaders[i]))
        target_iter = iter(self.target_loader)
        correct = 0

        for i in range(1, self.iteration+1):
            self.model.train()
            # LEARNING_RATE = self.lr / math.pow((1 + 10 * (i - 1) / (self.iteration)), 0.75)
            LEARNING_RATE = self.lr
            optimizer = torch.optim.Adam(
                self.model.parameters(), lr=LEARNING_RATE)

            for j in range(len(source_iters)):
                try:
                    source_data, source_label = next(source_iters[j])
                except Exception as err:
                    source_iters[j] = iter(self.source_loaders[j])
                    source_data, source_label = next(source_iters[j])
                try:
                    target_data, _ = next(target_iter)
                except Exception as err:
                    target_iter = iter(self.target_loader)
                    target_data, _ = next(target_iter)
                source_data, source_label = source_data.to(
                    device), source_label.to(device)
                target_data = target_data.to(device)

                optimizer.zero_grad()

                cls_loss, mmd_loss, disc_loss, lsd_loss = self.model(source_data, number_of_source=len(
                    source_iters), data_tgt=target_data, label_src=source_label, mark=j)
                alpha = 2 / (1 + math.exp(-10 * (i) / (self.iteration))) - 1
                beta = alpha/100
                gamma = alpha -1
                loss = cls_loss + alpha * mmd_loss + beta * disc_loss + gamma * lsd_loss
                loss.backward()
                optimizer.step()


            if i % (log_interval * 50) == 0:
                t_correct = self.test(i)
                if t_correct > correct:
                    correct = t_correct

        return 100. * correct / len(self.target_loader.dataset)

    def test(self, i):
        self.model.eval()
        test_loss = 0
        correct = 0
        corrects = []

        for i in range(len(self.source_loaders)):
            corrects.append(0)
        with torch.no_grad():
            for data, target in self.target_loader:
                data = data.to(device)
                target = target.to(device)
                preds = self.model(data, len(self.source_loaders))
                for i in range(len(preds)):
                    preds[i] = F.softmax(preds[i], dim=1)
                pred = sum(preds)/len(preds)
                test_loss += F.nll_loss(F.log_softmax(pred,
                                        dim=1), target.squeeze()).item()
                pred = pred.data.max(1)[1]
                correct += pred.eq(target.data.squeeze()).cpu().sum()


                for j in range(len(self.source_loaders)):
                    pred = preds[j].data.max(1)[1]
                    corrects[j] += pred.eq(target.data.squeeze()).cpu().sum()

            test_loss /= len(self.target_loader.dataset)
        return correct



def cross_subject(data, label, session_id, subject_id, category_number, batch_size, iteration, lr, momentum, log_interval,top_idxs):
    ## LOSO
    one_session_data, one_session_label = copy.deepcopy(data[session_id]), copy.deepcopy(label[session_id])
    test_idx = subject_id
    train_idxs = top_idxs
    target_data, target_label = copy.deepcopy(one_session_data[test_idx]), copy.deepcopy(one_session_label[test_idx])
    source_data, source_label = copy.deepcopy(one_session_data[train_idxs]), copy.deepcopy(one_session_label[train_idxs])
    print("selected_source_data.shape", source_data.shape)
    del one_session_label
    del one_session_data
    source_loaders = []
    for j in range(len(source_data)):
        source_loaders.append(torch.utils.data.DataLoader(dataset=utils.CustomDataset(source_data[j], source_label[j]),
                                                          batch_size=batch_size,
                                                          shuffle=True,
                                                          drop_last=True))
    target_loader = torch.utils.data.DataLoader(dataset=utils.CustomDataset(target_data, target_label),
                                                batch_size=batch_size,
                                                shuffle=True,
                                                drop_last=True)


    model = ASJDA(model=models.ASJDANet(pretrained=False, number_of_source=len(source_loaders),
                                            number_of_category=category_number),
                    source_loaders=source_loaders,
                    target_loader=target_loader,
                    batch_size=batch_size,
                    iteration=iteration,
                    lr=lr,
                    momentum=momentum,
                    log_interval=log_interval)
    # print(model.__getModel__())

    acc= model.train()
    print('Target_subject_id: {}, current_session_id: {}, acc: {}'.format(test_idx, session_id, acc))
    return acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ASJDA parameters')
    parser.add_argument('--dataset', type=str, default='seed3',
                        help='the dataset used for ASJDA, "seed3" or "seed4"')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='size for one batch, integer')
    parser.add_argument('--epoch', type=int, default=200,
                        help='training epoch, integer')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    args = parser.parse_args()
    dataset_name = args.dataset
    bn = args.norm_type

    # data preparation
    print('Model name: ASJDA. Dataset name: ', dataset_name)
    #seed and seed4
    data, label = utils.load_data(dataset_name)
    #deap
    #data, label = utils.load_deap(dataset_name)
    data_tmp = copy.deepcopy(data)
    label_tmp = copy.deepcopy(label)
    for i in range(len(data_tmp)):
        for j in range(len(data_tmp[0])):
            data_tmp[i][j] = utils.norminy(data_tmp[i][j])
    trial_total, category_number, _ = utils.get_number_of_label_n_trial(
        dataset_name)
    # training settings
    batch_size = args.batch_size
    epoch = args.epoch
    lr = args.lr
    print('BS: {}, epoch: {}'.format(batch_size, epoch))
    momentum = 0.9
    log_interval = 10
    iteration = 0
    if dataset_name == 'seed3':
        iteration = math.ceil(epoch * 3394 / batch_size)
    elif dataset_name == 'seed4':
        iteration = math.ceil(epoch * 820 / batch_size)
    elif dataset_name == 'deap':
        iteration = math.ceil(epoch * 2400 / batch_size)
    else:
        iteration = 5000
    print('Iteration: {}'.format(iteration))


    csub = []
    one_time=[]
    csub_all = []
    csesn = []
    select_threshold=[]

    t_values = [0.0045, 0.005, 0.0055, 0.0060]
    for session_id_main in range(3):
        for subject_id_main in range(15):
            #subject_id_main = 14
            js = []
            ks = []
            js_select_1 = []
            select_idxs1 = []
            sorted_tuples1 = []
            ks_select = []
            ks_select1 = []
            select_idxs_2 = []
            t_acc_pairs = []
            target_data, target_label = utils.get_one_data_and_label(data_tmp, label_tmp, session_id_main, subject_id_main)
            train_idxs = list(range(15))
            del train_idxs[subject_id_main]
            for value in train_idxs:
                source_data, source_label = utils.get_one_data_and_label(data_tmp, label_tmp, session_id_main,
                                                                     value)

                source_pdf = entropy(source_data, base=math.e)
                target_pdf= entropy(target_data, base=math.e)

                js_divergence = jensenshannon(source_pdf, target_pdf)
                js.append(js_divergence)

            # select threshold for current subject
            for threshold in t_values:
                print("threshold：", threshold)
                js_select_1=[]
                sorted_js1=[]
                min_values1=[]
                select_idxs1=[]
                for idx, js_divergence in enumerate(js):
                    if js_divergence < threshold:
                        select_idx= train_idxs[idx]
		                js_select_1.append((js_divergence, select_idx))
                if not js_select_1:
                    ks_select1 = [(js[i], train_idxs[i]) for i in range(len( js))]
	                sorted_js1 = sorted(ks_select1, key=lambda x: x[0])
	                min_values1 = sorted_js1[:7]
	                select_idxs1 = [item[1] for item in min_values1]
                else:
                    sorted_tuples1 = sorted(js_select_1, key=lambda x: x[0])
                    select_idxs1 = [tup[1] for tup in sorted_tuples1]


                test_idx = select_idxs1[0]
                del select_idxs1[0]
                train_idxs = select_idxs1
                print("test_idx", test_idx)
                print("train_idxs", train_idxs)
                acc1= cross_subject(data_tmp, label_tmp, session_id_main, test_idx,
                                                              category_number,
                                                              batch_size, iteration, lr, momentum, log_interval,
                                                              train_idxs)

                t_acc_pairs.append((threshold, acc1))

            t_max, max_acc = max(t_acc_pairs, key=lambda x: x[1])
            print("t_max, max_acc ",t_max, max_acc )
            select_threshold.append(t_max)

            # ---------start training---------------
            js_select = []
            sorted_tuples = []
            select_idxs=[]
            ks_select=[]
            for idx, js_divergence in enumerate(js):
                if js_divergence < t_max:
                    js_select.append((js_divergence, train_idxs[idx]))
            if not js_select:
                ks_select = [(js[i], train_idxs[i]) for i in range(len( js))]
	            sorted_js = sorted(ks_select, key=lambda x: x[0])
	            min_values = sorted_js[:7]
	            select_idxs = [item[1] for item in min_values]
            else:
                sorted_tuples = sorted(js_select, key=lambda x: x[0])
                select_idxs = [tup[1] for tup in sorted_tuples]

            acc= cross_subject(data_tmp, label_tmp, session_id_main, subject_id_main, category_number,
                                   batch_size, iteration, lr, momentum, log_interval, select_idxs)

            csub.append(acc)
            print("session:",session_id_main,"    traget_id:", subject_id_main,"   Cross-subject-acc: ", csub)

        one_session_all = csub
        sessions_data = {
            'acc': one_session_all,
            'threshold':select_threshold
        }

        scipy.io.savemat(f'ASJDA_{dataset_name}_sessions{session_id_main}_acc.mat', sessions_data)
        print("one_session_cross_subject_MEAN: ", np.mean(one_session_all), "std: ", np.std(one_session_all))
        one_session_all = []
        csub_all = csub
        csub = []

        print("-----------------                                                        -----------------")
        print("-----------------              one session finished                      -----------------")
        print("-----------------                                                        -----------------")


