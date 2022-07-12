import numpy as np
import os
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torch.optim as optim
import torch.nn.functional as F

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
import joblib

import argparse
import time
import pickle
import pandas as pd
from tqdm import tqdm
from collections import namedtuple, deque
import random

from dueling_dqn_model import DQN, Dueling_DQN, DuelingDQN, DuelingDQN_test, DuelingDQN_GRU, DuelingDQN_test_GRU, DuelingDQN_GRU_nomal, DuelingDQN_GRU_nomal_light, DuelingDQN_GRU_nomal_four, DuelingDQN_GRU_nomal_revise, DuelingDQN_GRU_revise, DuelingDQN_GRU_nomal_avt, DuelingDQN_GRU_nomal_try

from sklearn.metrics import f1_score, confusion_matrix, accuracy_score,\
                        classification_report, precision_recall_fscore_support, recall_score, precision_score

from dataloader_1 import IEMOCAPDataset, MELDDataset
from unimodel import uniaudiomodalpairs
from pair_datalodoader import IEMOCAP_pair_Dataset

OptimizerSpec = namedtuple("OptimizerSpec", ["constructor", "kwargs"])
# criterion = nn.MSELoss()
loss_weights = torch.FloatTensor([
                                    1/0.086747,
                                    1/0.144406,
                                    1/0.227883,
                                    1/0.160585,
                                    1/0.127711,
                                    1/0.252668,
                                    ])# the Proportion of each category of the database
torch.set_default_tensor_type(torch.FloatTensor)

# CUDA variables
USE_CUDA = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
dlongtype = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor

class LinearSchedule(object):
    def __init__(self, schedule_timesteps, final_p, initial_p=1.0):
        """Linear interpolation between initial_p and final_p over
        schedule_timesteps. After this many timesteps pass final_p is
        returned.
        Parameters
        ----------
        schedule_timesteps: int
            Number of timesteps for which to linearly anneal initial_p
            to final_p
        initial_p: float
            initial output value
        final_p: float
            final output value
        """
        self.schedule_timesteps = schedule_timesteps
        self.final_p            = final_p
        self.initial_p          = initial_p

    def value(self, t):
        """See Schedule.value"""
        fraction  = min(float(t) / self.schedule_timesteps, 1.0)
        return self.initial_p + fraction * (self.final_p - self.initial_p)

if __name__ == '__main__':
# def emotion_pair_dataframe_generalize(self, path):
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='does not use GPU')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate')
    parser.add_argument('--l2', type=float, default=0.00001, metavar='L2',
                        help='L2 regularization weight')
    parser.add_argument('--rec-dropout', type=float, default=0.1,
                        metavar='rec_dropout', help='rec_dropout rate')
    parser.add_argument('--dropout', type=float, default=0.1, metavar='dropout',
                        help='dropout rate')
    parser.add_argument('--batch-size', type=int, default=30, metavar='BS',
                        help='batch size')
    parser.add_argument('--epochs', type=int, default=60, metavar='E',
                        help='number of epochs')
    parser.add_argument('--class-weight', action='store_true', default=True,
                        help='class weight')
    parser.add_argument('--active-listener', action='store_true', default=False,
                        help='active listener')
    parser.add_argument('--attention', default='general', help='Attention type')
    parser.add_argument('--tensorboard', action='store_true', default=False,
                        help='Enables tensorboard log')
    parser.add_argument('--attribute', type=int, default=1, help='AVEC attribute')
    args = parser.parse_args()

    print(args)

    args.cuda = torch.cuda.is_available() and not args.no_cuda
    if args.cuda:
        print('Running on GPU')
    else:
        print('Running on CPU')

    n_classes  = 6
    cuda       = args.cuda
    n_epochs   = args.epochs
    numworkers = 0
    n_actions = ['0', '1', '2', '3', '4', '5']

    sum_iemocap = 0
    
    model_rf = RandomForestClassifier(n_estimators=2000)
    criterion = nn.CrossEntropyLoss(loss_weights.cuda() if cuda else loss_weights)

    test_label_pair_iemocap = []
    test_videoText_pair_iemocap = []
    test_videoAudio_pair_iemocap = []
    test_videoVisual_pair_iemocap = []

    epoch_num = 20
    batch = 10
    LEARNING_RATE = 0.00015
    ALPHA = 0.95
    EPS = 0.01
    exploration=LinearSchedule(1000000, 0.1)
    gamma = 0.9 #0.99
    target_update_freq = 100
    learning_freq = 4
    double_dqn = True
    num_param_updates = 0
    greedy = 0.95
    mu = 0
    sigma = 0.5

    cuda = 0
    device = torch.device("cuda:%d" % cuda if torch.cuda.is_available() else "cpu")

    testset_iemocap = IEMOCAPDataset(path='G:\\ins2\\AEPR\\IEMOCAP_features\\IEMOCAP_features_raw.pkl', train = False)

    Q = DuelingDQN(learning_rate = LEARNING_RATE,batch_size = batch)


    Q.load_state_dict(torch.load('Q.pkl'))
    Q.to(device)

    optimizer = OptimizerSpec(
        constructor=optim.Adam,
        kwargs=dict(lr=LEARNING_RATE, eps=EPS, weight_decay=0.00001)
    )
    optimizer = optimizer.constructor(Q.parameters(), **optimizer.kwargs)

    dir_path = "%s%d" % ('G:\\ins2\\AEPR', -1)
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    log_file = "%s/print.log" % dir_path
    f = open(log_file, "w+")
    sys.stdout = f

    domain = pd.read_pickle('G:\ins2\\AEPR\\knowdge_pair_iemocap.pkl')
    title = domain.pair_Labels
    p = domain.P
    
    with torch.no_grad():
        tot_num_t = 0.0
        tot_right_t = 0

        tot_0 = 0
        tot_0_r = 0
        tot_1 = 0
        tot_1_r = 0
        tot_2 = 0
        tot_2_r = 0
        tot_3 = 0
        tot_3_r = 0
        tot_4 = 0
        tot_4_r = 0
        tot_5 = 0
        tot_5_r = 0

        acc_0 = 0
        acc_1 = 0
        acc_2 = 0
        acc_3 = 0
        acc_4 = 0
        acc_5 = 0

        F_0 = 0
        F_1 = 0
        F_2 = 0
        F_3 = 0
        F_4 = 0
        F_5 = 0

        n = 0

        try_tot_revise = []
        try_tot_revise_real = []
        try_tot_real = []
        try_tot = []
        Y_valid = []
        action_r = []
        total_d = []
        total_action = []
        tot_right_try = 0
        tot_right_try_revise = 0
        tot_right_revise_max_real = 0
        tot_right_revise_real = 0
        tot_total = 0
        tot_total_acc = 0
        tot_total_recognition = 0

        with torch.no_grad():
        ###########################################################################################dialogue level test begin
            for idx in testset_iemocap.keys:
                lable_tem = testset_iemocap.videoLabels[idx]
                len_tem = len(lable_tem)
                title_tem = testset_iemocap.videoIDs[idx]
                videoacoustic_tem = testset_iemocap.videoAudio[idx]
                videovisual_tem = testset_iemocap.videoVisual[idx]
                videotext_tem = testset_iemocap.videoText[idx]

                j = int(0)
                n += 1

                visual = []
                audio = []
                text = []
                pair = []
                
                action_r = []
                try_tot_revise = []
                try_tot_revise_real = []
                try_tot_real = []
                try_tot = []
                Y_valid = []
                action_d = []
                recognition_action = []
                tot_right_try = 0
                tot_right_try_revise = 0
                tot_right_revise_max_real = 0
                tot_right_revise_real = 0

                Q.eval()

                for i in range(0, len_tem-3):
                    label_pair_tem = [lable_tem[i],lable_tem[i+1],lable_tem[i+2],lable_tem[i+3]]
                    videoacoustic_pair_tem = [videoacoustic_tem[i],videoacoustic_tem[i+1],videoacoustic_tem[i+2],videoacoustic_tem[i+3]]
                    videovisual_pair_tem = [videovisual_tem[i],videovisual_tem[i+1],videovisual_tem[i+2],videovisual_tem[i+3]]
                    videotext_pair_tem = [videotext_tem[i],videotext_tem[i+1],videotext_tem[i+2],videotext_tem[i+3]]

                    video_title_tem = title_tem[i+2]
                    video_correct_action_tem = lable_tem[i+3]

                    j += 1
                    
                    visual.append([videovisual_pair_tem])
                    audio.append([videoacoustic_pair_tem])
                    text.append([videotext_pair_tem])
                    pair.append([label_pair_tem])
                    action_d.append([video_correct_action_tem])

                visual = torch.tensor(visual)
                audio = torch.tensor(audio)
                text = torch.tensor(text)
                pair = torch.tensor(pair)
                action_d = torch.tensor(action_d)

                m = 50

                if j: #j>=m : for record the result of each step
                    for k in range(j):# m: for record the result of each step
                        states_f_text = text[k]
                        states_f_audio = audio[k]
                        states_f_visual = visual[k]
                        pair_Labels = pair[k]
                        action = action_d[k]
                        total_action.append(action)

                        states_f_text, states_f_audio, states_f_visual, pair_Labels, action = states_f_text.to(device), states_f_audio.to(device), states_f_visual.to(device), pair_Labels.to(device), action.to(device)
                        q_values = Q(states_f_text,states_f_audio,states_f_visual)
                        q_action = torch.argmax(q_values, dim = 1) # for dqn
                        q_action_t = F.softmax(q_values, dim = 1)

                        pair_Labels = torch.squeeze(pair_Labels) 
                        pair_Labels_g = pair_Labels.data.contiguous().view(-1) 
                        q_values = q_values.data.contiguous().view(4, -1) 
                        q_action_t = q_action_t.data.contiguous().view(4, -1) 
                        q_action = q_action.data.contiguous().view(-1) 

                        action = action.squeeze()
                        action_r.append(action)

                        Y_valid.append(action)

                        tem_pair = q_action
                        t_0 = tem_pair[0].tolist()
                        t_1 = tem_pair[1].tolist()
                        t_2 = tem_pair[2].tolist()

                        plus_weights = torch.FloatTensor([
                                            1.296747,
                                            1.074406,
                                            1.217883,
                                            1.180585,
                                            1.127711,
                                            1.182668,
                                            ]).to(device)

                        tem_pair_real = pair_Labels
                        t_0_real = tem_pair_real[0].tolist()
                        t_1_real = tem_pair_real[1].tolist()
                        t_2_real = tem_pair_real[2].tolist()

                        if k == 0:                       
                            t_real = str(t_0_real)+str(t_1_real)+str(t_2_real)
                        elif k == 1:
                            t_real = str(t_0_real)+str(t_1_real)+str(try_tot_revise_real[0].tolist())
                        elif k == 2:
                            t_real = str(t_0_real)+str(try_tot_revise_real[0].tolist())+str(try_tot_revise_real[1].tolist())
                        else:
                            t_real = str(try_tot_revise_real[k-3].tolist())+str(try_tot_revise_real[k-2].tolist())+str(try_tot_revise_real[k-1].tolist())

                        if t_real in title:
                            t_t_real = p[t_real]
                        else:
                            t_t_real = torch.ones(6)
                        t_t_real = torch.tensor(t_t_real)
                        t_t_real = torch.squeeze(t_t_real)
                        t_t_real = t_t_real.to(device)
                        t_max_real = torch.tensor(t_t_real)
                        t_max_real = torch.squeeze(t_max_real)
                        t_max_real = torch.argmax(t_max_real, dim = 0)
                        try_tot_real.append(t_max_real)
                        tem_t = q_action_t
                        t_revise_real = tem_t[3] # for ablation study 2021.02.27
                        # t_revise_real = tem_t[3] + 1.083*t_t_real
                        t_revise_real = torch.argmax(t_revise_real, dim = 0)
                        try_tot_revise_real.append(t_revise_real)
                        total_d.append(t_revise_real)

                        t = str(t_0)+str(t_1)+str(t_2)
                        if t in title:
                            t_t = p[t]
                        else:
                            t_t = torch.ones(6)
                        t_t = torch.tensor(t_t)
                        t_t = torch.squeeze(t_t)
                        t_t = t_t.to(device)
                        t_max = torch.tensor(t_t)
                        t_max = torch.squeeze(t_max)
                        t_max = torch.argmax(t_max, dim = 0)
                        try_tot.append(t_max)
                        tem_t = q_action_t
                        t_revise = tem_t[3] + plus_weights*t_t
                        t_revise = torch.argmax(t_revise, dim = 0)
                        try_tot_revise.append(t_revise)

                    tot_num_t = float(k+1)
                    tot_total += tot_num_t
                    Y_valid = torch.tensor(Y_valid)
                    Y_valid = torch.squeeze(Y_valid)
                    try_tot = torch.tensor(try_tot) 
                    tot_right_try += torch.sum(torch.eq(Y_valid, try_tot))# recognition library 
                    try_tot_revise = torch.tensor(try_tot_revise) 
                    tot_right_try_revise += torch.sum(torch.eq(Y_valid, try_tot_revise)) #recognition pair 
                    try_tot_revise_real = torch.tensor(try_tot_revise_real)
                    tot_right_revise_real += torch.sum(torch.eq(Y_valid, try_tot_revise_real)) #real pair
                    try_tot_real = torch.tensor(try_tot_real)
                    tot_right_revise_max_real += torch.sum(torch.eq(Y_valid, try_tot_real)) #library
                    tot_total_acc += tot_right_revise_real
                    tot_total_recognition += tot_right_try_revise

                    acc_try = tot_right_try/tot_num_t # recognition library max 
                    acc_try_revise = tot_right_try_revise/tot_num_t #recognition pair

                    acc_real = tot_right_revise_real/tot_num_t
                    acc_real_max = tot_right_revise_max_real/tot_num_t

                    true = (np.array(total_action)).astype(int)
                    pred = (np.array(total_d)).astype(int)

                    F_total = f1_score(true, pred, average='weighted')
                    F_0 = f1_score(true, pred, labels = [0], average='weighted')
                    F_1 = f1_score(true, pred, labels = [1], average='weighted')
                    F_2 = f1_score(true, pred, labels = [2], average='weighted')
                    F_3 = f1_score(true, pred, labels = [3], average='weighted')
                    F_4 = f1_score(true, pred, labels = [4], average='weighted')
                    F_5 = f1_score(true, pred, labels = [5], average='weighted')
                    recall_0 = recall_score(true, pred, labels = [0],average='micro')
                    recall_1 = recall_score(true, pred, labels = [1],average='micro')
                    recall_2 = recall_score(true, pred, labels = [2],average='micro')
                    recall_3 = recall_score(true, pred, labels = [3],average='micro')
                    recall_4 = recall_score(true, pred, labels = [4],average='micro')
                    recall_5 = recall_score(true, pred, labels = [5],average='micro')

                    precision_0 = precision_score(true, pred, labels = [0],average='macro')
                    precision_1 = precision_score(true, pred, labels = [1],average='macro')
                    precision_2 = precision_score(true, pred, labels = [2],average='macro')
                    precision_3 = precision_score(true, pred, labels = [3],average='macro')
                    precision_4 = precision_score(true, pred, labels = [4],average='macro')
                    precision_5 = precision_score(true, pred, labels = [5],average='macro')
                    acc_dilogue = accuracy_score(true, pred)
                    acc_total = tot_total_acc/tot_total
                    acc_rec = tot_total_recognition/tot_total
                    print('total num: %.4f total sum: %.4f' % (n, tot_total))
                    print('test length: %.4f' % (tot_num_t))
                    print('test acc_dilogue: %.4f acc_dialogue_rec: %.4f acc_total: %.4f acc_rec: %.4f' % (acc_real, acc_try_revise, acc_total, acc_rec))
                    print('test_real F: %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f' % (F_total, F_0, F_1, F_2, F_3, F_4, F_5))
                    print('test_real recall: %.4f, %.4f, %.4f, %.4f, %.4f, %.4f' % (recall_0, recall_1 , recall_2 , recall_3, recall_4, recall_5))
                    print('test_real precision: %.4f, %.4f, %.4f, %.4f, %.4f, %.4f' % (precision_0,precision_1, precision_2, precision_3, precision_4, precision_5))
                   

            ###########################################################################################dialogue level test end
            
    
    
