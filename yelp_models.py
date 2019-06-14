from __future__ import print_function
import torch
from torch.autograd import Variable
import torch.nn as nn
import os
import Constants
import torch
import torch.nn as nn
import torch.nn.functional as F

class BiLSTM(nn.Module):
    def __init__(self, config,):
        super(BiLSTM, self).__init__()
        self.embedding = nn.Embedding(config['n_vocab'], config['d_emb'], padding_idx=Constants.PAD)
        self.bilstm = nn.LSTM(config['d_emb'], config['d_hidden'], config['n_layers'], dropout=config['dropout'], bidirectional=True,batch_first=True)

    def forward(self, config, seq,hidden):
        embbed = F.relu(self.drop(self.embedding(seq))) # batch_size*max_len*embedding_size
        output = self.bilstm(embbed, hidden)[0]
        output = torch.max(output, 0)[0].squeeze()
        return output, embbed

class StructuredSelfAttention(torch.nn.Module):
    """implementation of the paper A Structured Self-Attentive Sentence Embedding
    """
    def __init__(self,config):
        super(StructuredSelfAttention,self).__init__()
        self.emb_dim=config["emb_dim"]
        self.vocab_size=config["vocab_size"]
        self.batch=config["batch_size"]
        self.hidden_dim=config["hidden_dim"]
        self.d_a=config["d_a"]
        self.r=config["r"]
        self.max_len=config["max_len"]
        self.n_classes=config["n_classes"]

        self.embedding = nn.Embedding(self.vocab_size, self.emb_dim, padding_idx=Constants.PAD)
        self.bilstm = nn.LSTM(self.emb_dim, self.hidden_dim, bidirectional=True,batch_first=True)
        self.fc1=nn.Linear(self.hidden_dim,self.n_classes)
        self.fc2=nn.Linear(self.d_a,self.r)
        self.fc3=nn.Linear(self.hidden_dim, self.n_classes)

        self.hidden_state = Variable(torch.randn(1,self.batch_size,self.hidden_dim)),Variable(torch.randn(1,self.batch_size,self.hidden_dim))

        def softmax(self, input, axis=1):
            """
            Softmax applied to axis=n
            Args:
               input: {Tensor,Variable} input on which softmax is to be applied
               axis : {int} axis on which softmax is to be applied
            Returns:
                softmaxed tensors

                @是用来对tensor进行矩阵相乘的：   cam_coords = (intrinsics_inv @ current_pixel_coords)
                *用来对tensor进行矩阵进行逐元素相乘：return cam_coords * depth.unsqueeze(1)
            """
            input_size = input.size()
            trans_input = input.transpose(axis, len(input_size) - 1)
            trans_size = trans_input.size()
            input_2d = trans_input.contiguous().view(-1, trans_size[-1])
            soft_max_2d = F.softmax(input_2d)
            soft_max_nd = soft_max_2d.view(*trans_size)

            return soft_max_nd.transpose(axis, len(input_size) - 1)

    def forward(self,x):
        embeded=self.embedding(x)
        outputs, self.hidden_state = self.lstm(embeded.view(self.batch_size, self.max_len, -1), self.hidden_state)
        x = F.tanh(self.fc1(outputs))
        x = self.fc2(x)
        x = self.softmax(x, 1)
        attention = x.transpose(1, 2)
        sentence_embeddings = attention @ outputs
        avg_sentence_embeddings = torch.sum(sentence_embeddings, 1) / self.r
        output = F.sigmoid(self.linear_final(avg_sentence_embeddings))
        return output, attention
    """
    output batch_size*r*2*hidden_dim
    """
