import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import tensorflow as tf


class DQN(nn.Module):
    def __init__(self, in_channels, num_actions):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)

        self.fc1 = nn.Linear(in_features=7*7*64, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=num_actions)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class Dueling_DQN(nn.Module):
    def __init__(self, in_channels, num_actions):
        super(Dueling_DQN, self).__init__()
        self.num_actions = num_actions
        
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)

        self.fc1_adv = nn.Linear(in_features=7*7*64, out_features=512)
        self.fc1_val = nn.Linear(in_features=7*7*64, out_features=512)

        self.fc2_adv = nn.Linear(in_features=512, out_features=num_actions)
        self.fc2_val = nn.Linear(in_features=512, out_features=1)

        self.relu = nn.ReLU()

    def forward(self, x):
        batch_size = x.size(0)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = x.view(x.size(0), -1)

        adv = self.relu(self.fc1_adv(x))
        val = self.relu(self.fc1_val(x))

        adv = self.fc2_adv(adv)
        val = self.fc2_val(val).expand(x.size(0), self.num_actions)
        
        x = val + adv - adv.mean(1).unsqueeze(1).expand(x.size(0), self.num_actions)
        return x

class Dueling_DQN_emotion_pair(nn.Module):
    def __init__(self, in_channels, num_actions):
        super(Dueling_DQN, self).__init__()
        self.num_actions = num_actions
        
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)

        self.fc1_adv = nn.Linear(in_features=7*7*64, out_features=512)
        self.fc1_val = nn.Linear(in_features=7*7*64, out_features=512)

        self.fc2_adv = nn.Linear(in_features=512, out_features=num_actions)
        self.fc2_val = nn.Linear(in_features=512, out_features=1)

        self.relu = nn.ReLU()

    def forward(self, states_t, states_a, states_v):
        batch_size = states_t.size(0)

        # a-t
        states_a = self.norma()

        # v-t

        #a-t-v
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = x.view(x.size(0), -1)

        adv = self.relu(self.fc1_adv(x))
        val = self.relu(self.fc1_val(x))

        adv = self.fc2_adv(adv)
        val = self.fc2_val(val).expand(x.size(0), self.num_actions)
        
        x = val + adv - adv.mean(1).unsqueeze(1).expand(x.size(0), self.num_actions)
        return x





class DuelingDQN(nn.Module):
    def __init__(
            self,
            n_actions=6,
            n_features=100,
            learning_rate=0.001,
            reward_decay=0.9,
            e_greedy=0.9,
            replace_target_iter=200,
            memory_size=500,
            batch_size=32,
            e_greedy_increment=None,
            output_graph=False,
            dueling=True,
            sess=None,
    ):
        super(DuelingDQN, self).__init__()
        self.n_actions = n_actions
        self.n_features = n_features # each modal of each timestep length = 100
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max
        self.dropProb = 0.4#0.4 #0.6
        self.num_layer = 1
        self.num = 512

        self.dueling = dueling      # decide to use dueling DQN or not

        self.learn_step_counter = 0
        self.memory = np.zeros((self.memory_size, n_features*4*6+2)) # two states with 3 modals and 4 timesteps

        self.norm_uni = nn.BatchNorm1d(n_features)
        self.norm_uni_v = nn.BatchNorm1d(512)
        self.drop = nn.Dropout(p=self.dropProb)
        self.line_uni = nn.Linear(n_features, 512)
        self.line_uni_v = nn.Linear(512, self.num)
        self.lstm_uni = nn.LSTM(self.num,self.num,num_layers=self.num_layer, dropout = self.dropProb, bidirectional = True, batch_first = True)
        self.line_bi = nn.Linear((self.num*2*2),self.num)
        self.line_mul = nn.Linear((self.num*2),self.num*4)
        self.norm_mul = nn.BatchNorm1d(self.num*4)
        self.lstm_mul = nn.LSTM(self.num,self.num,num_layers=self.num_layer, dropout = self.dropProb, bidirectional = True, batch_first = True)

        self.fc1_adv = nn.Linear(in_features=4*self.num*2, out_features=512)
        self.fc1_val = nn.Linear(in_features=4*self.num*2, out_features=512)

        self.fc2_adv = nn.Linear(in_features=self.num, out_features=self.n_actions)
        self.fc2_val = nn.Linear(in_features=self.num, out_features=1)

        self.relu = nn.ReLU()
        self.att = nn.Linear(self.num*2,1,bias=False)


    def forward(self, states_t, states_a, states_v):
        states_t = torch.squeeze(states_t)
        states_a = torch.squeeze(states_a)
        states_v = torch.squeeze(states_v)

        t = states_v.size()

        batch = states_t.size()[0]
        seq = states_t.size()[1]
        cuda = 0
        device = torch.device("cuda:%d" % cuda if torch.cuda.is_available() else "cpu")

        states_t = states_t.data.contiguous().view(batch, -1, seq)
        states_a = states_a.data.contiguous().view(batch, -1, seq)
        states_v = states_v.data.contiguous().view(batch, -1, seq)
        batch_l = torch.tensor(batch)
        batch_l = batch_l.to(device)

        states_t = self.norm_uni(states_t)
        states_a = self.norm_uni(states_a)
        states_v = self.norm_uni_v(states_v)

        states_t = states_t.data.contiguous().view(batch, seq, -1)
        states_t_in = self.line_uni(self.drop(states_t))
        states_t_out,_ = self.lstm_uni(states_t_in)#[batch,4,512*2]
        # states_t_out = self.drop(states_t_out)#new added
        # states_t_out = torch.cat([torch.zeros(batch, 1, 512*2).to(device), states_t_out], 1)#[batch, 4+1, 512*2]
        # batch_l = batch_l.data.contiguous().view(-1)#[10]
        # states_t_Selector = torch.zeros(batch, 1, 4 + 1).to(device).scatter_(2, batch_l.unsqueeze(0).unsqueeze(0), 1.0)#[batch,1,4+1]
        states_t_out_M = states_t_out.permute(1,0,2)
        states_t_out_M = self.att(states_t_out_M)# seq_len, batch, 1
        states_t_Selector = F.softmax(states_t_out_M, dim=0).permute(1,2,0)# batch, 1, seq_len
        states_t_State = torch.matmul(states_t_Selector, states_t_out).squeeze()#[batch,512*2]

        states_a = states_a.data.contiguous().view(batch, seq, -1)
        states_a_in = self.line_uni(self.drop(states_a))
        states_a_out,_ = self.lstm_uni(states_a_in)
        # states_a_out = self.drop(states_a_out) #new added
        # states_a_out = torch.cat([torch.zeros(batch, 1, 512*2).to(device), states_a_out], 1)#[batch, 4+1, 512*2]
        # batch_l = batch_l.data.contiguous().view(-1)#[10]
        # states_a_Selector = torch.zeros(batch, 1, 4 + 1).to(device).scatter_(2, batch_l.unsqueeze(1).unsqueeze(1), 1.0)#[batch,1,4+1]
        states_a_out_M = states_a_out.permute(1,0,2)
        states_a_out_M = self.att(states_a_out_M)# seq_len, batch, 1
        states_a_Selector = F.softmax(states_a_out_M, dim=0).permute(1,2,0)# batch, 1, seq_len
        states_a_State = torch.matmul(states_a_Selector, states_a_out).squeeze()#[batch,512*2]

        states_v = states_v.data.contiguous().view(batch, seq, -1)
        states_v_in = self.line_uni_v(self.drop(states_v))
        states_v_out,_ = self.lstm_uni(states_v_in)
        # states_v_out = self.drop(states_v_out)#new added
        # states_v_out = torch.cat([torch.zeros(batch, 1, 512*2).to(device), states_v_out], 1)#[batch, 4+1, 512*2]
        # batch_l = batch_l.data.contiguous().view(-1)#[10]
        # states_v_Selector = torch.zeros(batch, 1, 4 + 1).to(device).scatter_(2, batch_l.unsqueeze(1).unsqueeze(1), 1.0)#[batch,1,4+1]
        states_v_out_M = states_v_out.permute(1,0,2)
        states_v_out_M = self.att(states_v_out_M)# seq_len, batch, 1
        states_v_Selector = F.softmax(states_v_out_M, dim=0).permute(1,2,0)# batch, 1, seq_len
        states_v_State = torch.matmul(states_v_Selector, states_v_out).squeeze()#[batch,512*2]

        states_at = self.line_bi(torch.cat([states_a_State, states_t_State], 1))
        states_vt = self.line_bi(torch.cat([states_v_State, states_t_State], 1))
        states_avt = self.line_mul(torch.cat([states_at, states_vt], 1))
        states_avt = self.drop(self.norm_mul(states_avt))#[batch,512*4]
        states_avt = states_avt.data.contiguous().view(batch,-1,512)#[batch,4,512]

        output,_ = self.lstm_mul(states_avt)#[batch,4,512*2]
        output = output.data.contiguous().view(batch,-1)#[batch,4*512*2]
        # output = self.drop(output)#new added

        adv = self.relu(self.fc1_adv(output))#[batch,512]
        val = self.relu(self.fc1_val(output))#[batch,512]

        adv = self.fc2_adv(adv)#[batch,6]
        val = self.fc2_val(val).expand(output.size(0), self.n_actions)
        
        output = val + adv - adv.mean(1).unsqueeze(1).expand(output.size(0), self.n_actions)
        
        return output


class DuelingDQN_test(nn.Module):
    def __init__(
            self,
            n_actions=6,
            n_features=100,
            learning_rate=0.001,
            reward_decay=0.9,
            e_greedy=0.9,
            replace_target_iter=200,
            memory_size=500,
            batch_size=32,
            e_greedy_increment=None,
            output_graph=False,
            dueling=True,
            sess=None,
    ):
        super(DuelingDQN_test, self).__init__()
        self.n_actions = n_actions
        self.n_features = n_features # each modal of each timestep length = 100
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max
        self.dropProb = 0.5 #0.6
        self.num_layer = 1
        self.num_base = 512

        self.dueling = dueling      # decide to use dueling DQN or not

        self.learn_step_counter = 0
        self.memory = np.zeros((self.memory_size, n_features*4*6+2)) # two states with 3 modals and 4 timesteps

        self.norm_uni = nn.BatchNorm1d(n_features)
        self.norm_uni_v = nn.BatchNorm1d(512)
        self.drop = nn.Dropout(p=self.dropProb)
        self.line_uni = nn.Linear(n_features, self.num_base)
        self.line_uni_v = nn.Linear(512, self.num_base*2)
        self.lstm_uni = nn.LSTM(self.num_base,self.num_base,num_layers=self.num_layer, dropout = self.dropProb, bidirectional = True, batch_first = True)
        self.lstm_uni_v = nn.LSTM(self.num_base*2,self.num_base*4,num_layers=self.num_layer, dropout = self.dropProb, bidirectional = True, batch_first = True)
        self.line_bi = nn.Linear((self.num_base*2*2),self.num_base)
        self.line_mul = nn.Linear((self.num_base*2),self.num_base*4)
        self.norm_mul = nn.BatchNorm1d(self.num_base*4)
        self.lstm_mul = nn.LSTM(self.num_base,self.num_base,num_layers=self.num_layer, dropout = self.dropProb, bidirectional = True, batch_first = True)

        self.fc1_adv = nn.Linear(in_features=4*512*2, out_features=512)
        self.fc1_val = nn.Linear(in_features=4*512*2, out_features=512)

        self.fc2_adv = nn.Linear(in_features=512, out_features=self.n_actions)
        self.fc2_val = nn.Linear(in_features=512, out_features=1)

        self.relu = nn.ReLU()
        self.att = nn.Linear(self.num_base*2,1,bias=False)
        self.att_v = nn.Linear(self.num_base*4*2,1,bias=False)
        self.line_v = nn.Linear(self.num_base*4*2,self.num_base*2)


    def forward(self, states_t, states_a, states_v):
        states_t = torch.squeeze(states_t)
        states_a = torch.squeeze(states_a)
        states_v = torch.squeeze(states_v)

        t = states_v.size()

        batch = states_t.size()[0]
        seq = states_t.size()[1]
        cuda = 0
        device = torch.device("cuda:%d" % cuda if torch.cuda.is_available() else "cpu")

        states_t = states_t.data.contiguous().view(batch, -1, seq)
        states_a = states_a.data.contiguous().view(batch, -1, seq)
        states_v = states_v.data.contiguous().view(batch, -1, seq)
        batch_l = torch.tensor(batch)
        batch_l = batch_l.to(device)

        states_t = self.norm_uni(states_t)
        states_a = self.norm_uni(states_a)
        states_v = self.norm_uni_v(states_v)

        states_t = states_t.data.contiguous().view(batch, seq, -1)
        states_t_in = self.line_uni(self.drop(states_t))
        states_t_out,_ = self.lstm_uni(states_t_in)#[batch,4,512*2]
        # states_t_out = self.drop(states_t_out)#new added
        # states_t_out = torch.cat([torch.zeros(batch, 1, 512*2).to(device), states_t_out], 1)#[batch, 4+1, 512*2]
        # batch_l = batch_l.data.contiguous().view(-1)#[10]
        # states_t_Selector = torch.zeros(batch, 1, 4 + 1).to(device).scatter_(2, batch_l.unsqueeze(0).unsqueeze(0), 1.0)#[batch,1,4+1]
        states_t_out_M = states_t_out.permute(1,0,2)
        states_t_out_M = self.att(states_t_out_M)# seq_len, batch, 1
        states_t_Selector = F.softmax(states_t_out_M, dim=0).permute(1,2,0)# batch, 1, seq_len
        states_t_State = torch.matmul(states_t_Selector, states_t_out).squeeze()#[batch,512*2]

        states_a = states_a.data.contiguous().view(batch, seq, -1)
        states_a_in = self.line_uni(self.drop(states_a))
        states_a_out,_ = self.lstm_uni(states_a_in)
        # states_a_out = self.drop(states_a_out) #new added
        # states_a_out = torch.cat([torch.zeros(batch, 1, 512*2).to(device), states_a_out], 1)#[batch, 4+1, 512*2]
        # batch_l = batch_l.data.contiguous().view(-1)#[10]
        # states_a_Selector = torch.zeros(batch, 1, 4 + 1).to(device).scatter_(2, batch_l.unsqueeze(1).unsqueeze(1), 1.0)#[batch,1,4+1]
        states_a_out_M = states_a_out.permute(1,0,2)
        states_a_out_M = self.att(states_a_out_M)# seq_len, batch, 1
        states_a_Selector = F.softmax(states_a_out_M, dim=0).permute(1,2,0)# batch, 1, seq_len
        states_a_State = torch.matmul(states_a_Selector, states_a_out).squeeze()#[batch,512*2]

        states_v = states_v.data.contiguous().view(batch, seq, -1)
        states_v_in = self.line_uni_v(self.drop(states_v))
        states_v_out,_ = self.lstm_uni_v(states_v_in)
        # states_v_out = self.drop(states_v_out)#new added
        # states_v_out = torch.cat([torch.zeros(batch, 1, 512*2).to(device), states_v_out], 1)#[batch, 4+1, 512*2]
        # batch_l = batch_l.data.contiguous().view(-1)#[10]
        # states_v_Selector = torch.zeros(batch, 1, 4 + 1).to(device).scatter_(2, batch_l.unsqueeze(1).unsqueeze(1), 1.0)#[batch,1,4+1]
        states_v_out_M = states_v_out.permute(1,0,2)
        states_v_out_M = self.att_v(states_v_out_M)# seq_len, batch, 1
        states_v_Selector = F.softmax(states_v_out_M, dim=0).permute(1,2,0)# batch, 1, seq_len
        states_v_State = torch.matmul(states_v_Selector, states_v_out).squeeze()#[batch,512*4*2]
        states_v_State = self.line_v(states_v_State)

        states_at = self.line_bi(torch.cat([states_a_State, states_t_State], 1))
        states_vt = self.line_bi(torch.cat([states_v_State, states_t_State], 1))
        states_avt = self.line_mul(torch.cat([states_at, states_vt], 1))
        states_avt = self.drop(self.norm_mul(states_avt))#[batch,512*4]
        states_avt = states_avt.data.contiguous().view(batch,-1,512)#[batch,4,512]

        output,_ = self.lstm_mul(states_avt)#[batch,4,512*2]
        output = output.data.contiguous().view(batch,-1)#[batch,4*512*2]
        # output = self.drop(output)#new added

        adv = self.relu(self.fc1_adv(output))#[batch,512]
        val = self.relu(self.fc1_val(output))#[batch,512]

        adv = self.fc2_adv(adv)#[batch,6]
        val = self.fc2_val(val).expand(output.size(0), self.n_actions)
        
        output = val + adv - adv.mean(1).unsqueeze(1).expand(output.size(0), self.n_actions)
        
        return output


class DuelingDQN(nn.Module):
    def __init__(
            self,
            n_actions=6,
            n_features=100,
            learning_rate=0.001,
            reward_decay=0.9,
            e_greedy=0.9,
            replace_target_iter=200,
            memory_size=500,
            batch_size=32,
            e_greedy_increment=None,
            output_graph=False,
            dueling=True,
            sess=None,
    ):
        super(DuelingDQN, self).__init__()
        self.n_actions = n_actions
        self.n_features = n_features # each modal of each timestep length = 100
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max
        self.dropProb = 0.4 #0.6
        self.num_layer = 1

        self.dueling = dueling      # decide to use dueling DQN or not

        self.learn_step_counter = 0
        self.memory = np.zeros((self.memory_size, n_features*4*6+2)) # two states with 3 modals and 4 timesteps

        self.norm_uni = nn.BatchNorm1d(n_features)
        self.norm_uni_v = nn.BatchNorm1d(512)
        self.norm_uni_4d = nn.BatchNorm2d(n_features, track_running_stats=False)
        self.norm_uni_v_4d = nn.BatchNorm2d(512, track_running_stats=False)
        self.drop = nn.Dropout(p=self.dropProb)
        self.line_uni = nn.Linear(n_features, 512)
        self.line_uni_v = nn.Linear(512, 512)
        self.lstm_uni = nn.LSTM(512,512,num_layers=self.num_layer, dropout = self.dropProb, bidirectional = True, batch_first = True)
        self.line_bi = nn.Linear((512*2*2),512)
        self.line_mul = nn.Linear((512*2),512*4)
        self.norm_mul = nn.BatchNorm1d(512*4)
        self.lstm_mul = nn.LSTM(512,512,num_layers=self.num_layer, dropout = self.dropProb, bidirectional = True, batch_first = True)
        self.gru_uni = nn.GRU(512,512,num_layers=self.num_layer, dropout = self.dropProb, bidirectional = True, batch_first = True)
        self.gru_mul = nn.GRU(512,512,num_layers=self.num_layer, dropout = self.dropProb, bidirectional = True, batch_first = True)

        self.fc1_adv = nn.Linear(in_features=4*512*2, out_features=512)
        self.fc1_val = nn.Linear(in_features=4*512*2, out_features=512)

        self.fc2_adv = nn.Linear(in_features=512, out_features=self.n_actions)
        self.fc2_val = nn.Linear(in_features=512, out_features=1)

        self.relu = nn.ReLU()
        self.att = nn.Linear(512*2,1,bias=False)


    def forward(self, states_t, states_a, states_v):
        # states_t = torch.squeeze(states_t)
        # states_a = torch.squeeze(states_a)
        # states_v = torch.squeeze(states_v)

        t = states_v.size()

        batch = states_t.size()[0]
        seq = states_t.size()[2]
        cuda = 0
        device = torch.device("cuda:%d" % cuda if torch.cuda.is_available() else "cpu")

        states_t = states_t.data.contiguous().view(batch, -1, 1, seq)# batch, channel, 1, 4
        states_a = states_a.data.contiguous().view(batch, -1, 1, seq)
        states_v = states_v.data.contiguous().view(batch, -1, 1, seq)
        batch_l = torch.tensor(batch)
        batch_l = batch_l.to(device)

        states_t = self.norm_uni_4d(states_t)
        states_a = self.norm_uni_4d(states_a)
        states_v = self.norm_uni_v_4d(states_v)

        states_t = torch.squeeze(states_t)# batch, channel, 4
        states_a = torch.squeeze(states_a)
        states_v = torch.squeeze(states_v)

        # states_t = states_t.data.contiguous().view(batch, -1, seq)
        # states_a = states_a.data.contiguous().view(batch, -1, seq)
        # states_v = states_v.data.contiguous().view(batch, -1, seq)
        # batch_l = torch.tensor(batch)
        # batch_l = batch_l.to(device)

        # states_t = self.norm_uni(states_t)
        # states_a = self.norm_uni(states_a)
        # states_v = self.norm_uni_v(states_v)

        states_t = states_t.data.contiguous().view(batch, seq, -1)
        states_t_in = self.line_uni(self.drop(states_t))
        states_t_out,_ = self.gru_uni(states_t_in)#[batch,4,512*2]
        # states_t_out = self.drop(states_t_out)#new added
        # states_t_out = torch.cat([torch.zeros(batch, 1, 512*2).to(device), states_t_out], 1)#[batch, 4+1, 512*2]
        # batch_l = batch_l.data.contiguous().view(-1)#[10]
        # states_t_Selector = torch.zeros(batch, 1, 4 + 1).to(device).scatter_(2, batch_l.unsqueeze(0).unsqueeze(0), 1.0)#[batch,1,4+1]
        states_t_out_M = states_t_out.permute(1,0,2)
        states_t_out_M = self.att(states_t_out_M)# seq_len, batch, 1
        states_t_Selector = F.softmax(states_t_out_M, dim=0).permute(1,2,0)# batch, 1, seq_len
        states_t_State = torch.matmul(states_t_Selector, states_t_out).squeeze()#[batch,512*2]

        states_a = states_a.data.contiguous().view(batch, seq, -1)
        states_a_in = self.line_uni(self.drop(states_a))
        states_a_out,_ = self.gru_uni(states_a_in)
        # states_a_out = self.drop(states_a_out) #new added
        # states_a_out = torch.cat([torch.zeros(batch, 1, 512*2).to(device), states_a_out], 1)#[batch, 4+1, 512*2]
        # batch_l = batch_l.data.contiguous().view(-1)#[10]
        # states_a_Selector = torch.zeros(batch, 1, 4 + 1).to(device).scatter_(2, batch_l.unsqueeze(1).unsqueeze(1), 1.0)#[batch,1,4+1]
        states_a_out_M = states_a_out.permute(1,0,2)
        states_a_out_M = self.att(states_a_out_M)# seq_len, batch, 1
        states_a_Selector = F.softmax(states_a_out_M, dim=0).permute(1,2,0)# batch, 1, seq_len
        states_a_State = torch.matmul(states_a_Selector, states_a_out).squeeze()#[batch,512*2]

        states_v = states_v.data.contiguous().view(batch, seq, -1)
        states_v_in = self.line_uni_v(self.drop(states_v))
        states_v_out,_ = self.gru_uni(states_v_in)
        # states_v_out = self.drop(states_v_out)#new added
        # states_v_out = torch.cat([torch.zeros(batch, 1, 512*2).to(device), states_v_out], 1)#[batch, 4+1, 512*2]
        # batch_l = batch_l.data.contiguous().view(-1)#[10]
        # states_v_Selector = torch.zeros(batch, 1, 4 + 1).to(device).scatter_(2, batch_l.unsqueeze(1).unsqueeze(1), 1.0)#[batch,1,4+1]
        states_v_out_M = states_v_out.permute(1,0,2)
        states_v_out_M = self.att(states_v_out_M)# seq_len, batch, 1
        states_v_Selector = F.softmax(states_v_out_M, dim=0).permute(1,2,0)# batch, 1, seq_len
        states_v_State = torch.matmul(states_v_Selector, states_v_out).squeeze()#[batch,512*2]

        states_at = self.line_bi(torch.cat([states_a_State, states_t_State], 1))
        states_vt = self.line_bi(torch.cat([states_v_State, states_t_State], 1))
        states_avt = self.line_mul(torch.cat([states_at, states_vt], 1))
        states_avt = self.drop(self.norm_mul(states_avt))#[batch,512*4]
        states_avt = states_avt.data.contiguous().view(batch,-1,512)#[batch,4,512]

        output,_ = self.gru_mul(states_avt)#[batch,4,512*2]
        output = output.data.contiguous().view(batch,-1)#[batch,4*512*2]
        # output = self.drop(output)#new added

        adv = self.relu(self.fc1_adv(output))#[batch,512]
        val = self.relu(self.fc1_val(output))#[batch,512]

        adv = self.fc2_adv(adv)#[batch,6]
        val = self.fc2_val(val).expand(output.size(0), self.n_actions)
        
        output = val + adv - adv.mean(1).unsqueeze(1).expand(output.size(0), self.n_actions)
        
        return output


