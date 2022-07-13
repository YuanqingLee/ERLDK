import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torch.optim as optim
import torch.nn.functional as F

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

import argparse
import pandas as pd
from tqdm import tqdm
from collections import namedtuple, deque
import random

from dueling_dqn_model import DQN, Dueling_DQN, DuelingDQN, DuelingDQN_test, DuelingDQN_GRU, DuelingDQN_test_GRU, DuelingDQN_GRU_nomal, DuelingDQN_GRU_nomal_light, DuelingDQN_GRU_nomal_four, DuelingDQN_GRU_nomal_revise, DuelingDQN_GRU_revise, DuelingDQN_GRU_nomal_avt, DuelingDQN_GRU_nomal_try

from sklearn.metrics import f1_score, confusion_matrix, accuracy_score,\
                        classification_report, precision_recall_fscore_support, recall_score, precision_score

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
                                    ])
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
    
    # model_rf = RandomForestClassifier(n_estimators=2000)
    criterion = nn.CrossEntropyLoss(loss_weights.cuda() if cuda else loss_weights)

    test_label_pair_iemocap = []
    test_videoText_pair_iemocap = []
    test_videoAudio_pair_iemocap = []
    test_videoVisual_pair_iemocap = []

    epoch_num = 5
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

    ########################
    Q = DuelingDQN(learning_rate = LEARNING_RATE,batch_size = batch)
    Q_target = DuelingDQN(learning_rate = LEARNING_RATE,batch_size = batch)
    ########################

    Q.to(device)
    Q_target.to(device)

    optimizer = OptimizerSpec(
        constructor=optim.Adam,
        kwargs=dict(lr=LEARNING_RATE, eps=EPS, weight_decay=0.00001)
    )
    optimizer = optimizer.constructor(Q.parameters(), **optimizer.kwargs)

    ##############################begin train#####################################################
    pair_env_train = IEMOCAP_pair_Dataset(path='...\\trainset_pair_new_justtrain_4.pkl')
    train_loader = DataLoader(
                                dataset=pair_env_train,
                                batch_size=batch,
                                shuffle=True,
                                num_workers=0,
    )
    pair_env_test = IEMOCAP_pair_Dataset(path='...\\testset_pair_new_justtest_4.pkl')
    test_loader = DataLoader(
                                dataset=pair_env_test,
                                batch_size=10,
                                shuffle=False,
                                num_workers=0,
    )
    domain = pd.read_pickle('...\\knowdge_pair_iemocap.pkl')
    title = domain.pair_Labels
    p = domain.P
    
    acc_best = 0
    for epoch in tqdm(range(epoch_num)):
        print("now the epoch number is: %d" % epoch)
        tot_num = 0
        tot_right = 0

        # with grad
        for i, data in enumerate(train_loader):
            states_titles, pair_Labels, states_f_text, states_f_audio, states_f_visual, next_states_titles, next_states_f_text, next_states_f_audio, next_states_f_visual, action, done = data
            observations = data
            batch_tem = len(action)
            done = done.squeeze()
            pair_Labels, states_f_text, states_f_audio, states_f_visual, next_states_f_text, next_states_f_audio, next_states_f_visual, action, done = pair_Labels.to(
                device), states_f_text.to(device), states_f_audio.to(device), states_f_visual.to(device), next_states_f_text.to(device), next_states_f_audio.to(
                    device), next_states_f_visual.to(device), action.to(device), done.to(device)

            ####################### without exploration
            Q.train()
            q_values = Q(states_f_text,states_f_audio,states_f_visual)
            q_action = torch.argmax(q_values, dim = 1) # for dqn
            # q_action = F.log_softmax(q_values, dim = 1) # for nomal without dqn
            # q_action_t = F.softmax(q_values, dim = 1) # for nomal without dqn
            # q_action = torch.argmax(q_action, dim = 1) # for nomal without dqn
            q_action = F.log_softmax(q_values, dim = 1) # for 4-pair nomal without dqn
            q_action_t = F.softmax(q_values, dim = 1) # for 4-pair nomal without dqn
            q_action = torch.argmax(q_action, dim = 1) # for 4-pair nomal without dqn
            pair_Labels_g = torch.squeeze(pair_Labels) # for 4-pair nomal without dqn
            pair_Labels_g = pair_Labels_g.data.contiguous().view(-1) # for 4-pair nomal without dqn
            # ####################### without exploration

            ######################################################################nomal no dqn
            ############################# backwards pass
            # action_g = torch.squeeze(action) # for nomal without dqn
            optimizer.zero_grad()
            # loss = criterion(q_values, action_g) # for nomal without dqn
            loss = criterion(q_values, pair_Labels_g) # for 4-pair nomal without dqn
            loss.backward()
            # print(loss)

            ############################ for nomal without dqn
            # for j in range(batch_tem):
            #     if q_action[j] == action[j]:
            #         tot_right +=1
            # tot_num += batch_tem
            ############################ for nomal without dqn

            ############################ for 4-pair nomal without dqn
            for j in range(batch_tem*4):
                if q_action[j] == pair_Labels_g[j]:
                    tot_right +=1
            tot_num += batch_tem*4
            ############################ for 4-pair nomal without dqn
            ######################################################################nomal no dqn

            #######################exploration
            sample = random.random()
            threshold = exploration.value(i)
            q_values = Q(states_f_text,states_f_audio,states_f_visual)
            if np.random.uniform() > greedy:
                q_action = torch.argmax(q_values, dim = 1)
            else:
                q_action = torch.LongTensor(np.random.randint(0,6,size = 10))
                q_action = q_action.to(device)
            #######################exploration

            ##############################################################################dqn
            q_s_a = q_values.gather(1, q_action.unsqueeze(1))
            q_s_a = q_s_a.squeeze()

            reward = []
        
            for j in range(batch_tem):
                if q_action[j] == action[j]:
                    reward.extend([2.0])
                    tot_right +=1
                else:
                    reward.extend([-2.0])
            tot_num += batch_tem

            reward = np.array(reward)
            
            # clipping the reward, noted in nature paper
            # reward = np.clip(reward, -1.0, 1.0)
            reward = torch.from_numpy(reward)
            reward = reward.to(device)

            if (double_dqn):
                # ---------------
                #   double DQN
                # ---------------

                # get the Q values for best actions in obs_tp1 
                # based off the current Q network
                # max(Q(s', a', theta_i)) wrt a'
                q_tp1_values = Q(next_states_f_text,next_states_f_audio,next_states_f_visual).detach()
                _, a_prime = q_tp1_values.max(1)
                # a_prime = torch.max(q_tp1_values, 1)[0]
                # a_prime = torch.argmax(q_tp1_values, dim = 1)

                # get Q values from frozen network for next state and chosen action
                # Q(s',argmax(Q(s',a', theta_i), theta_i_frozen)) (argmax wrt a')
                q_target_tp1_values = Q_target(next_states_f_text,next_states_f_audio,next_states_f_visual).detach()
                q_target_s_a_prime = q_target_tp1_values.gather(1, a_prime.unsqueeze(1))
                q_target_s_a_prime = q_target_s_a_prime.squeeze()

                # if current state is end of episode, then there is no next Q value
                q_target_s_a_prime = (1 - done) * q_target_s_a_prime 

                expected_Q = reward + gamma * q_target_s_a_prime
            else:
                # ---------------
                #   regular DQN
                # ---------------

                # get the Q values for best actions in obs_tp1 
                # based off frozen Q network
                # max(Q(s', a', theta_i_frozen)) wrt a'
                q_tp1_values = Q_target(next_states_f_text,next_states_f_audio,next_states_f_visual).detach()
                q_s_a_prime, a_prime = q_tp1_values.max(1)

                # if current state is end of episode, then there is no next Q value
                q_s_a_prime = (1 - done) * q_s_a_prime 

                # Compute Bellman error
                # r + gamma * Q(s',a', theta_i_frozen) - Q(s, a, theta_i)
                expected_Q = reward + gamma * q_s_a_prime

            # clip the error and flip 
            # clipped_error = -1.0 * error.clamp(-1, 1)

            # backwards pass
            optimizer.zero_grad()
            q_s_a = q_s_a.type(torch.float64)
            error = expected_Q - q_s_a
            # clipped_error = -1.0 * error.clamp(-1, 1)
            clipped_error = -1.0 * error
            # loss = criterion(q_s_a, expected_Q)
            # loss.backward()
            q_s_a.backward(clipped_error)
            ##############################################################################dqn

            # update
            optimizer.step()
            num_param_updates += 1

            # update target Q network weights with current Q network weights
            # if num_param_updates % target_update_freq == 0:
            #     Q_target.load_state_dict(Q.state_dict())
        
        print("now the epoch number is: %d" % epoch)
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

        try_tot_revise = []
        try_tot_revise_real = []
        try_tot_real = []
        try_tot = []
        Y_valid = []
        action_r = []
        tot_right_try = 0
        tot_right_try_revise = 0
        tot_right_revise_max_real = 0
        tot_right_revise_real = 0

        with torch.no_grad():
            for i, data in enumerate(test_loader):
                states_titles, pair_Labels, states_f_text, states_f_audio, states_f_visual, next_states_titles, next_states_f_text, next_states_f_audio, next_states_f_visual, action, done = data
                observations = data
                done = done.squeeze()
                batch_tem = len(action)

                pair_Labels, states_f_text, states_f_audio, states_f_visual, next_states_f_text, next_states_f_audio, next_states_f_visual, action, done = pair_Labels.to(
                    device), states_f_text.to(device), states_f_audio.to(device), states_f_visual.to(device), next_states_f_text.to(device), next_states_f_audio.to(
                        device), next_states_f_visual.to(device), action.to(device), done.to(device)
                
                if states_f_visual.size()[0] == 1:
                    continue
                
                Q.eval()
                q_values = Q(states_f_text,states_f_audio,states_f_visual)
                # q_action = torch.argmax(q_values, dim = 1) # for dqn
                # q_action = F.log_softmax(q_values, dim = 1) # for nomal without dqn
                # q_action = torch.argmax(q_action, dim = 1) # for nomal without dqn
                q_action = F.log_softmax(q_values, dim = 1) # for 4-pair nomal without dqn
                q_action_t = F.softmax(q_values, dim = 1)# for 4-pair nomal without dqn
                q_action = torch.argmax(q_action, dim = 1) # for 4-pair nomal without dqn

                pair_Labels = torch.squeeze(pair_Labels) # for both without dqn
                pair_Labels_g = pair_Labels_g.data.contiguous().view(-1) # for 4-pair nomal without dqn
                q_values = q_values.data.contiguous().view(batch_tem, 4, -1) # for 4-pair nomal without dqn
                q_action_t = q_action_t.data.contiguous().view(batch_tem, 4, -1) # for 4-pair nomal without dqn
                q_action = q_action.data.contiguous().view(batch_tem, -1) # for 4-pair nomal without dqn

                action = action.squeeze()
                action_r.extend(action)
                tot_num_t += len(action)
                # tot_num += batch_tem*4

                for j in range(batch_tem):
                    tem = q_values[j]
                    Y_valid.append(action[j])

                    ################################### for 4-pair nomal without dqn
                    tem_pair = q_action[j]
                    t_0 = tem_pair[0].tolist()
                    t_1 = tem_pair[1].tolist()
                    t_2 = tem_pair[2].tolist()
                    ################################### for 4-pair nomal without dqn

                    tem_pair_real = pair_Labels[j]
                    t_0_real = tem_pair_real[0].tolist()
                    t_1_real = tem_pair_real[1].tolist()
                    t_2_real = tem_pair_real[2].tolist()

                    t_real = str(t_0_real)+str(t_1_real)+str(t_2_real)
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
                    tem_t = q_action_t[j]
                    # t_revise_real = tem_t + 1.5*t_t_real # for nomal without dqn
                    t_revise_real = tem_t[3] + 1.5*t_t_real # for 4-pair nomal without dqn
                    t_revise_real = torch.argmax(t_revise_real, dim = 0)
                    try_tot_revise_real.append(t_revise_real)

                    ################################### for 4-pair nomal without dqn
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
                    tem_t = q_action_t[j]
                    t_revise = tem_t[3] * t_t
                    t_revise = torch.argmax(t_revise, dim = 0)
                    try_tot_revise.append(t_revise)
                    ################################### for 4-pair nomal without dqn

            for k in range(len(action_r)):
                if try_tot_revise_real[k] == 0:
                    tot_0 += 1
                elif try_tot_revise_real[k] == 1:
                    tot_1 +=1
                elif try_tot_revise_real[k] == 2:
                    tot_2 +=1
                elif try_tot_revise_real[k] == 3:
                    tot_3 +=1
                elif try_tot_revise_real[k] == 4:
                    tot_4 +=1
                elif try_tot_revise_real[k] == 5:
                    tot_5 +=1

            for k in range(len(action_r)):
                if action_r[k] == 0:
                    acc_0 += 1
                elif action_r[k] == 1:
                    acc_1 +=1
                elif action_r[k] == 2:
                    acc_2 +=1
                elif action_r[k] == 3:
                    acc_3 +=1
                elif action_r[k] == 4:
                    acc_4 +=1
                elif action_r[k] == 5:
                    acc_5 +=1
        
            for j in range(len(action_r)):
                tem_pair = try_tot_revise_real[j]
                if tem_pair == action_r[j]: # for 4-pair nomal without dqn
                # if tem_pair == action[j]: # for nomal without dqn
                    tot_right_t +=1
                    if action_r[j] == 0:
                        tot_0_r += 1
                    elif action_r[j] == 1:
                        tot_1_r +=1
                    elif action_r[j] == 2:
                        tot_2_r +=1
                    elif action_r[j] == 3:
                        tot_3_r +=1
                    elif action_r[j] == 4:
                        tot_4_r +=1
                    elif action_r[j] == 5:
                        tot_5_r +=1
            
            true = (np.array(action_r)).astype(int)
            pred = (np.array(try_tot_revise_real)).astype(int)

            recall = tot_right_t/tot_num_t
            acc_total = accuracy_score(true, pred)
            # recall_0 = tot_0_r/tot_0
            # recall_1 = tot_1_r/tot_1
            # recall_2 = tot_2_r/tot_2
            # recall_3 = tot_3_r/tot_3
            # recall_4 = tot_4_r/tot_4
            # recall_5 = tot_5_r/tot_5
            recall_0 = recall_score(true, pred, labels = [0],average='micro')
            recall_1 = recall_score(true, pred, labels = [1],average='micro')
            recall_2 = recall_score(true, pred, labels = [2],average='micro')
            recall_3 = recall_score(true, pred, labels = [3],average='micro')
            recall_4 = recall_score(true, pred, labels = [4],average='micro')
            recall_5 = recall_score(true, pred, labels = [5],average='micro')

            precision = tot_right/tot_num
            # precision_0 = tot_0_r/acc_0
            # precision_1 = tot_1_r/acc_1
            # precision_2 = tot_2_r/acc_2
            # precision_3 = tot_3_r/acc_3
            # precision_4 = tot_4_r/acc_4
            # precision_5 = tot_5_r/acc_5
            precision_0 = precision_score(true, pred, labels = [0],average='macro')
            precision_1 = precision_score(true, pred, labels = [1],average='macro')
            precision_2 = precision_score(true, pred, labels = [2],average='macro')
            precision_3 = precision_score(true, pred, labels = [3],average='macro')
            precision_4 = precision_score(true, pred, labels = [4],average='macro')
            precision_5 = precision_score(true, pred, labels = [5],average='macro')

            # F_0 = (2*precision_0*recall_0)/(precision_0+recall_0)
            # F_1 = (2*precision_0*recall_1)/(precision_0+recall_1)
            # F_2 = (2*precision_0*recall_2)/(precision_0+recall_2)
            # F_3 = (2*precision_0*recall_3)/(precision_0+recall_3)
            # F_4 = (2*precision_0*recall_4)/(precision_0+recall_4)
            # F_5 = (2*precision_0*recall_5)/(precision_0+recall_5)
            F_total = f1_score(true, pred, average='weighted')
            F_0 = f1_score(true, pred, labels = [0], average='weighted')
            F_1 = f1_score(true, pred, labels = [1], average='weighted')
            F_2 = f1_score(true, pred, labels = [2], average='weighted')
            F_3 = f1_score(true, pred, labels = [3], average='weighted')
            F_4 = f1_score(true, pred, labels = [4], average='weighted')
            F_5 = f1_score(true, pred, labels = [5], average='weighted')

            Y_valid = torch.tensor(Y_valid)
            Y_valid = torch.squeeze(Y_valid)
            try_tot = torch.tensor(try_tot) # for 4-pair nomal without dqn
            tot_right_try += torch.sum(torch.eq(Y_valid, try_tot))# recognition library # for 4-pair nomal without dqn
            try_tot_revise = torch.tensor(try_tot_revise) # for 4-pair nomal without dqn
            tot_right_try_revise += torch.sum(torch.eq(Y_valid, try_tot_revise)) #recognition pair # for 4-pair nomal without dqn
            try_tot_revise_real = torch.tensor(try_tot_revise_real)
            tot_right_revise_real += torch.sum(torch.eq(Y_valid, try_tot_revise_real)) #real pair
            try_tot_real = torch.tensor(try_tot_real)
            tot_right_revise_max_real += torch.sum(torch.eq(Y_valid, try_tot_real)) #library

            acc_r_t = tot_right_t/tot_num_t # own
            acc_try = tot_right_try/tot_num_t # recognition library max # for 4-pair nomal without dqn
            acc_try_revise = tot_right_try_revise/tot_num_t #recognition pair # for 4-pair nomal without dqn

            acc_real = tot_right_revise_real/tot_num_t
            acc_real_max = tot_right_revise_max_real/tot_num_t
            if acc_r_t > acc_best:
                acc_best = acc_r_t
            print('test [%d, %5d] acc_real: %.4f acc_real_best:%.4f acc_recognition: %.4f' % (epoch + 1, i + 1, acc_r_t, acc_best, acc_try_revise))
            print('test_real [%d, %5d] recall: %.4f, %.4f, %.4f, %.4f, %.4f, %.4f' % (epoch + 1, i + 1, recall_0, recall_1 , recall_2 , recall_3, recall_4, recall_5))
            print('test_real [%d, %5d] precision: %.4f, %.4f, %.4f, %.4f, %.4f, %.4f' % (epoch + 1, i + 1, precision_0,precision_1, precision_2, precision_3, precision_4, precision_5))
            print('test_real [%d, %5d] F: %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f' % (epoch + 1, i + 1, F_total, F_0, F_1, F_2, F_3, F_4, F_5))
        
        acc = tot_right/tot_num
        print('train [%d, %5d] acc: %.3f' % (epoch + 1, i + 1, (tot_right/tot_num)))
    
    # torch.save(Q.state_dict(), 'Q.pkl')
    ##############################end train#####################################################
    
   
