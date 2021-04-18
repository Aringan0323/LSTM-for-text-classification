import numpy as np
import pickle as pkl
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import *
torch.manual_seed(0)
c_len = 100
embed_d = 128

# data class
class input_data():
    def load_text_data(self, word_n = 100000):
        f = open('../dataset/paper_abstract.pkl', 'rb')
        p_content_set = pkl.load(f)
        f.close()

        p_label = [0] * 21044
        label_f = open('../dataset/paper_label.txt', 'r')
        for line in label_f:
            line = line.strip()
            label_s = re.split('\t',line)
            p_label[int(label_s[0])] = int(label_s[1])
        label_f.close()

        def remove_unk(x):
            return [[1 if w >= word_n else w for w in sen] for sen in x]

        p_content, p_content_id = p_content_set
        p_content = remove_unk(p_content)

        # padding with max len
        for i in range(len(p_content)):
            if len(p_content[i]) > c_len:
                p_content[i] = p_content[i][:c_len]
            else:
                pad_len = c_len - len(p_content[i])
                p_content[i] = np.lib.pad(p_content[i], (0, pad_len), 'constant', constant_values=(0,0))

        p_id_train = []
        p_content_train = []
        p_label_train = []
        p_id_test = []
        p_content_test = []
        p_label_test = []
        for j in range(len(p_content)):
            if j % 10 in (3, 6, 9):
                p_id_test.append(p_content_id[j])
                #p_content_test.append(p_content[j])
                p_label_test.append(p_label[j])
            else:
                p_id_train.append(p_content_id[j])
                #p_content_train.append(p_content[j])
                p_label_train.append(p_label[j])

        p_train_set = (p_id_train, p_label_train)
        p_test_set = (p_id_test, p_label_test)

        return p_content, p_train_set, p_test_set


    def load_word_embed(self):
        word_embed = np.zeros((32786, 128))
        len = word_embed.shape[0]

        word_embed[0:int(len/4)] = np.genfromtxt('../dataset/word_embeddings1.txt', delimiter=' ')
        word_embed[int(len/4):int(len/2)] = np.genfromtxt('../dataset/word_embeddings2.txt', delimiter=' ')
        word_embed[int(len/2):int(3*len/4)] = np.genfromtxt('../dataset/word_embeddings3.txt', delimiter=' ')
        word_embed[int(3*len/4):] = np.genfromtxt('../dataset/word_embeddings4.txt', delimiter=' ')
        return word_embed



#text RNN Encoder
class Text_Encoder(nn.Module):
    def __init__(self, p_content, word_embed, mean_pooling=False, device=torch.device('cpu')):
        # two input: p_content - abstract data of all papers, word_embed - pre-trained word embedding
        super(Text_Encoder, self).__init__()
        self.p_content = p_content
        self.word_embed = word_embed
        self.mean_pooling = mean_pooling
        self.device = device
        self.lstm = nn.LSTM(input_size=128, hidden_size=64, batch_first=True)
        self.fchl = nn.Linear(64, 32)
        self.fcol = nn.Linear(32, 5)


    def forward(self, id_batch):
        relu = nn.ReLU()
        sigmoid = nn.Sigmoid()

        # id_batch: use id_batch (paper ids in this batch) to obtain paper conent of this batch
        x = torch.zeros((id_batch.shape[0], 100, 128))
        for i in range(id_batch.shape[0]):
            content = self.p_content[id_batch[i]]
            for j in range(content.shape[0]):
                x[i,j] = self.word_embed[int(content[j])]

        x = x.to(self.device)

        output, (h_n, c_n) = self.lstm(x)

        if self.mean_pooling:
            num_layers = h_n.shape[0]
            h_mean = torch.zeros(h_n.shape[1:])
            for i in range(num_layers):
                h_mean = h_mean + h_n[i]
            h_mean = torch.div(h_mean, num_layers)
            x = h_mean
        else:
            x = h_n[-1]
        x = relu(x)
        x = sigmoid(self.fchl(x))
        x = self.fcol(x)
        return x
