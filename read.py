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
import tqdm

from sklearn.metrics import f1_score, confusion_matrix, accuracy_score,\
                        classification_report, precision_recall_fscore_support

from dataloader_1 import IEMOCAPDataset, MELDDataset

def get_IEMOCAP_loaders(path, batch_size, valid=0.1, num_workers=0, pin_memory=False):

    batch_size = sum_iemocap

    train_loader = DataLoader(trainset_pair_iemocap,
                              batch_size=batch_size,
                            #   sampler=train_sampler,
                            #   collate_fn=trainset_iemocap.collate_fn,
                              num_workers=num_workers)

    # train_loader = DataLoader(trainset_iemocap,
    #                           batch_size=batch_size,
    #                         #   sampler=train_sampler,
    #                           collate_fn=trainset_iemocap.collate_fn,
    #                           num_workers=num_workers,
    #                           pin_memory=pin_memory)
    # valid_loader = DataLoader(trainset,
    #                           batch_size=batch_size,
    #                         #   sampler=valid_sampler,
    #                           collate_fn=trainset.collate_fn,
    #                           num_workers=num_workers,
    #                           pin_memory=pin_memory)

    # testset = IEMOCAPDataset(path=path, train=False)
    # test_loader = DataLoader(testset,
    #                          batch_size=batch_size,
    #                          collate_fn=testset.collate_fn,
    #                          num_workers=num_workers,
    #                          pin_memory=pin_memory)

    return train_loader
    # , valid_loader, test_loader

def get_MELD_loaders(path, n_classes, batch_size=32, valid=0.1, num_workers=0, pin_memory=False):
    trainset_meld = MELDDataset(path=path, n_classes=n_classes)
    # train_sampler, valid_sampler = get_train_valid_sampler(trainset, valid)
    train_loader = DataLoader(trainset_meld,
                              batch_size=batch_size,
                            #   sampler=train_sampler,
                              collate_fn=trainset_meld.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)
    # valid_loader = DataLoader(trainset_meld,
    #                           batch_size=batch_size,
    #                           sampler=valid_sampler,
    #                           collate_fn=trainset_meld.collate_fn,
    #                           num_workers=num_workers,
    #                           pin_memory=pin_memory)

    # testset = MELDDataset(path=path, n_classes=n_classes, train=False)
    # test_loader = DataLoader(trainset_meld,
    #                          batch_size=batch_size,
    #                          collate_fn=trainset_meld.collate_fn,
    #                          num_workers=num_workers,
    #                          pin_memory=pin_memory)

    return train_loader
    # , valid_loader, test_loader

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

    sum_iemocap = 0
    sum_meld = 0
    sum_daily_train = 0
    sum_daily_test = 0
    sum_daily_valid = 0
    
    # label_pair_meld = []

    train_label_pair_iemocap = []
    train_videoText_pair_iemocap = []
    train_videoAudio_pair_iemocap = []
    train_videoVisual_pair_iemocap = []

    test_label_pair_iemocap = []
    test_videoText_pair_iemocap = []
    test_videoAudio_pair_iemocap = []
    test_videoVisual_pair_iemocap = []

    trainset_iemocap = IEMOCAPDataset(path='...\\IEMOCAP_features_raw.pkl')
    testset_iemocap = IEMOCAPDataset(path='...\\IEMOCAP_features_raw.pkl', train = False)

    trainset_pair_iemocap = pd.DataFrame(columns=('states_titles', 'pair_Labels', 'states_f_text', 'states_f_audio', 'states_f_visual', 
                                                  'next_states_titles', 'next_states_f_text', 'next_states_f_audio', 'next_states_f_visual', 
                                                  'action', 'done'))
    
    testset_pair_iemocap = pd.DataFrame(columns=('states_titles', 'pair_Labels', 'states_f_text', 'states_f_audio', 'states_f_visual', 
                                                  'next_states_titles', 'next_states_f_text', 'next_states_f_audio', 'next_states_f_visual', 
                                                  'action', 'done'))

    dir_path = "%s%d" % ('...', -1)
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    log_file = "%s/print.log" % dir_path
    f = open(log_file, "w+")
    sys.stdout = f

    # for IEMOCAP train
    for idx in trainset_iemocap.keys:

        lable_tem = trainset_iemocap.videoLabels[idx]
        len_tem = len(lable_tem)
        title_tem = trainset_iemocap.videoIDs[idx]
        videoacoustic_tem = trainset_iemocap.videoAudio[idx]
        videovisual_tem = trainset_iemocap.videoVisual[idx]
        videotext_tem = trainset_iemocap.videoText[idx]

        for i in range(0, len_tem-3, 1):
            label_pair_tem = [lable_tem[i],lable_tem[i+1],lable_tem[i+2],lable_tem[i+3]]
            videoacoustic_pair_tem = [videoacoustic_tem[i],videoacoustic_tem[i+1],videoacoustic_tem[i+2],videoacoustic_tem[i+3]]
            videovisual_pair_tem = [videovisual_tem[i],videovisual_tem[i+1],videovisual_tem[i+2],videovisual_tem[i+3]]
            videotext_pair_tem = [videotext_tem[i],videotext_tem[i+1],videotext_tem[i+2],videotext_tem[i+3]]

            video_title_tem = title_tem[i+2]
            video_correct_action_tem = lable_tem[i+3]

            if i == (len_tem-4):
                videoacoustic_pair_next_tem = videoacoustic_pair_tem # the next state features will be recorded as itself.
                videovisual_pair_next_tem = videovisual_pair_tem
                videotext_pair_next_tem = videotext_pair_tem
                video_done_tem = 1 # this current dialogue has finished without the next state
                video_title_next_tem = 'no_next_states'
            else:
                videoacoustic_pair_next_tem = [videoacoustic_tem[i+1],videoacoustic_tem[i+2],videoacoustic_tem[i+3],videoacoustic_tem[i+4]]
                videovisual_pair_next_tem = [videovisual_tem[i+1],videovisual_tem[i+2],videovisual_tem[i+3],videovisual_tem[i+4]]
                videotext_pair_next_tem = [videotext_tem[i+1],videotext_tem[i+2],videotext_tem[i+3],videotext_tem[i+4]]
                video_done_tem = 0 # this current dialogue has not finished with a next state
                video_title_next_tem = title_tem[i+3]

            trainset_pair_iemocap = trainset_pair_iemocap.append({'states_titles': video_title_tem, 'pair_Labels': [label_pair_tem],
                            'states_f_text': [videotext_pair_tem], 'states_f_audio': [videoacoustic_pair_tem], 'states_f_visual': [videovisual_pair_tem], 
                            'next_states_titles': video_title_next_tem, 
                            'next_states_f_text': [videotext_pair_next_tem], 'next_states_f_audio': [videoacoustic_pair_next_tem], 'next_states_f_visual': [videovisual_pair_next_tem],
                            'action': [video_correct_action_tem], 'done': [video_done_tem]}, ignore_index=True)
            
    # trainset_pair_iemocap.to_pickle('trainset_preprocess_pair.pkl')

    # pair_env_train = pd.read_pickle('D:\\LYQ\\ins2\\AEPR\\trainset_preprocess_pair.pkl')
    trainset_pair_iemocap.index = pd.Series(trainset_pair_iemocap.states_titles)

    trainset_pair_iemocap.to_pickle('trainset_pair.pkl') # 3: step = 2 4: step = 1


    # for IEMOCAP test pair
    for idx in testset_iemocap.keys:

        lable_tem = testset_iemocap.videoLabels[idx]
        len_tem = len(lable_tem)
        title_tem = testset_iemocap.videoIDs[idx]
        videoacoustic_tem = testset_iemocap.videoAudio[idx]
        videovisual_tem = testset_iemocap.videoVisual[idx]
        videotext_tem = testset_iemocap.videoText[idx]

        for i in range(0, len_tem-3, 1):
            label_pair_tem = [lable_tem[i],lable_tem[i+1],lable_tem[i+2],lable_tem[i+3]]
            videoacoustic_pair_tem = [videoacoustic_tem[i],videoacoustic_tem[i+1],videoacoustic_tem[i+2],videoacoustic_tem[i+3]]
            videovisual_pair_tem = [videovisual_tem[i],videovisual_tem[i+1],videovisual_tem[i+2],videovisual_tem[i+3]]
            videotext_pair_tem = [videotext_tem[i],videotext_tem[i+1],videotext_tem[i+2],videotext_tem[i+3]]

            video_title_tem = title_tem[i+2]
            video_correct_action_tem = lable_tem[i+3]

            if i == (len_tem-4): #and (len_tem%4) == 0:#i == (len_tem-(len_tem%4)-4):
                videoacoustic_pair_next_tem = videoacoustic_pair_tem # the next state features will be recorded as itself.
                videovisual_pair_next_tem = videovisual_pair_tem
                videotext_pair_next_tem = videotext_pair_tem
                video_done_tem = 1 # this current dialogue has finished without the next state
                video_title_next_tem = 'no_next_states'
            else:
                videoacoustic_pair_next_tem = [videoacoustic_tem[i+1],videoacoustic_tem[i+2],videoacoustic_tem[i+3],videoacoustic_tem[i+4]]
                videovisual_pair_next_tem = [videovisual_tem[i+1],videovisual_tem[i+2],videovisual_tem[i+3],videovisual_tem[i+4]]
                videotext_pair_next_tem = [videotext_tem[i+1],videotext_tem[i+2],videotext_tem[i+3],videotext_tem[i+4]]
                video_done_tem = 0 # this current dialogue has not finished with a next state
                video_title_next_tem = title_tem[i+3]

            testset_pair_iemocap = testset_pair_iemocap.append({'states_titles': video_title_tem, 'pair_Labels': [label_pair_tem],
                            'states_f_text': [videotext_pair_tem], 'states_f_audio': [videoacoustic_pair_tem], 'states_f_visual': [videovisual_pair_tem], 
                            'next_states_titles': video_title_next_tem, 
                            'next_states_f_text': [videotext_pair_next_tem], 'next_states_f_audio': [videoacoustic_pair_next_tem], 'next_states_f_visual': [videovisual_pair_next_tem],
                            'action': [video_correct_action_tem], 'done': [video_done_tem]}, ignore_index=True)

    testset_pair_iemocap.index = pd.Series(testset_pair_iemocap.states_titles)

    testset_pair_iemocap.to_pickle('testset_pair.pkl')




    
    
