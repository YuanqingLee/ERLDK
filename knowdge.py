import numpy as np
import os
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torch.optim as optim

import argparse
import time
import pickle
import pandas as pd
from tqdm import tqdm
from collections import namedtuple, deque

from sklearn.metrics import f1_score, confusion_matrix, accuracy_score,\
                        classification_report, precision_recall_fscore_support

from dataloader_1 import IEMOCAPDataset

if __name__ == '__main__':
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
    batch = 10

    sum_iemocap = 0

    knowdge_pair_iemocap = pd.DataFrame(columns=('pair_Labels', 'P'))
    
    dir_path = "%s%d" % ('D:\\LYQ\\ins2\\AEPR', -1)
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    log_file = "%s/print.log" % dir_path
    f = open(log_file, "w+")
    sys.stdout = f

    trainset_iemocap = IEMOCAPDataset(path='D:\\LYQ\\ins2\\AEPR\\IEMOCAP_features\\IEMOCAP_features_raw.pkl') # with full daset key
    pair_labels = []

    for idx in trainset_iemocap.keys:
        lable_tem = trainset_iemocap.videoLabels[idx]
        len_tem = len(lable_tem)
        title_tem = trainset_iemocap.videoIDs[idx]
        videoacoustic_tem = trainset_iemocap.videoAudio[idx]
        videovisual_tem = trainset_iemocap.videoVisual[idx]
        videotext_tem = trainset_iemocap.videoText[idx]

        for i in range(0, len_tem-6, 1):
            label_pair_tem = [lable_tem[i],lable_tem[i+1],lable_tem[i+2],lable_tem[i+3],lable_tem[i+4],lable_tem[i+5]]
            pair_labels.append(label_pair_tem)
    
    label = pair_labels
    len_label = len(label)

    for j in range(6):
        for k in range(6):
            for m in range(6):
                for n in range(6):
                    for o in range(6):
                        videotext_pair_tem = []
                        tot_0 = 0
                        tot_1 = 0
                        tot_2 = 0
                        tot_3 = 0
                        tot_4 = 0
                        tot_5 = 0

                        tot = 0
                        for q in range(len_label):
                            label_tem = label[q]
                            l = label_tem
                            if (l[0] == j) and (l[1] == k) and (l[2] == m) and (l[3] == n) and (l[4] == o):
                                if (l[5] == 0):
                                    tot_0 +=1
                                elif (l[5] == 1):
                                    tot_1 +=1
                                elif (l[5] == 2):
                                    tot_2 +=1
                                elif (l[5] == 3):
                                    tot_3 +=1
                                elif (l[5] == 4):
                                    tot_4 +=1
                                else:
                                    tot_5 +=1
                                
                                tot += 1
                        if tot == 0:
                            continue
                        label_pair_tem = str(j)+str(k)+str(m)+str(n)+str(o)
                        videotext_pair_tem.append(tot_0/tot)
                        videotext_pair_tem.append(tot_1/tot)
                        videotext_pair_tem.append(tot_2/tot)
                        videotext_pair_tem.append(tot_3/tot)
                        videotext_pair_tem.append(tot_4/tot)
                        videotext_pair_tem.append(tot_5/tot)
                        knowdge_pair_iemocap = knowdge_pair_iemocap.append({'pair_Labels': label_pair_tem,'P': [videotext_pair_tem]}, ignore_index=True)
                        print('%s %.5f %.5f %.5f %.5f %.5f %.5f' % (label_pair_tem, (tot_0/tot), (tot_1/tot), (tot_2/tot), (tot_3/tot), (tot_4/tot), (tot_5/tot)))

    knowdge_pair_iemocap.index = pd.Series(knowdge_pair_iemocap.pair_Labels)
    knowdge_pair_iemocap.to_pickle('knowdge_pair_iemocap_5.pkl')
