import torch
from torch import nn
from Util import *
import torch.nn.functional as F
import math

dtype = torch.FloatTensor


class CNNEncoder(nn.Module):
    """docstring for ClassName"""

    def __init__(self, vocab_size, embed_dim, weights):
        super(CNNEncoder, self).__init__()
        # self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.embedding = nn.Embedding(vocab_size, embed_dim).from_pretrained(weights)
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=0),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=0),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            nn.ReLU())

    def forward(self, x):
        x = self.embedding(x)  # support:5*1*28*28 batch:95*1*28*28  ->5*1*28*300
        out = self.layer1(x)  # 5*64*13*13  #5*64*13*129
        out = self.layer2(out)  # 5*64*5*5
        out = self.layer3(out)  # 5*64*5*5
        out = self.layer4(out)  # 5*64*5*5
        # out = x
        out = out.view(out.size(0), -1)
        return out  # 64？  #5*8400


class RelationNetwork(nn.Module):
    """docstring for RelationNetwork"""

    def __init__(self, input_size, hidden_size):
        super(RelationNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        torch.nn.init.xavier_uniform(self.fc1.weight)

        self.fc2 = nn.Linear(hidden_size, 1)
        torch.nn.init.xavier_uniform(self.fc2.weight)

        self.fc3 = nn.Linear(input_size, 1)
        torch.nn.init.xavier_uniform(self.fc3.weight)

        self.dropout = torch.nn.Dropout(0.1)

    def forward(self, x):  # 475*100  #valid 25*100
        out = x
        out = out.view(out.size(0), -1)  # [475,100]  #valid 25*100

        out = F.relu(self.fc1(self.dropout(out)))  # [475,8]  #25*8
        out = F.sigmoid(self.fc2(self.dropout(out)))  # [475,1]  #valid 25*1

        out = self.fc3(self.dropout(x))
        out = F.sigmoid(out)  # 直接映射
        return out


class RelationNetwork0(nn.Module):
    """docstring for RelationNetwork"""

    def __init__(self, input_size, hidden_size):
        super(RelationNetwork, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x):  # 475*128*5*5
        # out = self.layer1(x)  # [475,64,2,2]
        # out = self.layer2(out)  # [475,64,1,1]
        out = x
        out = out.view(out.size(0), -1)  # [475,64]
        out = F.relu(self.fc1(out))  # [475,8]
        out = F.sigmoid(self.fc2(out))  # [475,1]
        return out


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm') != -1:
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        n = m.weight.size(1)
        # m.weight.data.normal_(0, 0.01)
        m.weight.data.normal_(0, 0.01)
        m.bias.data = torch.ones(m.bias.data.size())


class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, n_filters, filter_sizes, hidden_size,
                 dropout, pad_idx):
        super().__init__()
        self.filter_sizes = filter_sizes
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=Constants.PAD)
        # self.embedding.weight.data.uniform_(-1, 1)
        # self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=Constants.PAD)
        # self.covlayer=nn.ModuleList
        self.convs = [nn.Conv2d(in_channels=1, out_channels=n_filters,
                                kernel_size=(filter_size, embed_dim), bias=True)
                      for filter_size in filter_sizes]  # [3,4,5]
        # self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)
        # self.dropout = nn.Dropout(dropout)

    def forward(self, text):  # batch_size*seq_len
        seq_len = text.shape[1]
        # text = [sent len, batch size]
        # text = text.permute(1, 0)
        # text = [batch size, sent len]
        embedded = self.embedding(text)  # batch_size*se1_len*emb_dim
        # embedded = [batch size, sent len, emb dim]
        embedded = embedded.unsqueeze(1)  # batch_size*1*seq_len#emb_dim
        # embedded = [batch size, 1, sent len, emb dim]
        pooled = []
        for i in range(len(self.filter_sizes)):
            conved = F.relu(self.convs[i](embedded))  # batch_size*out_channels*(seq_len-2)*1
            conved = F.max_pool2d(conved, (seq_len - self.filter_sizes[i] + 1, 1))  # batch_size*out_channels*1*1
            pooled.append(conved)

        # conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]  # batch_size*n_filters*[18,17,16]
        # # conved_n = [batch size, n_filters, sent len - filter_sizes[n] + 1]
        # pooled = [F.max_pool2d(conv,()) for conv in conved]  # 提取最显著的词
        # pooled_n = [batch size, n_filters]
        # cat = self.dropout(torch.cat(pooled, dim=1)).squeeze(-1).squeeze(-1)
        # cat = [batch size, n_filters * len(filter_sizes)]
        cat = torch.cat(pooled, dim=1).squeeze(-1).squeeze(-1)
        return cat


class TEXTCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, n_filters, filter_sizes, output_dim,
                 dropout, pad_idx):
        super().__init__()
        self.filter_sizes = filter_sizes
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=Constants.PAD)
        # self.embedding.weight.data.uniform_(-1, 1)
        # self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=Constants.PAD)
        # self.covlayer=nn.ModuleList
        self.convs = [nn.Conv2d(in_channels=1, out_channels=n_filters,
                                kernel_size=(filter_size, embed_dim), bias=True)
                      for filter_size in filter_sizes]  # [3,4,5]
        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):  # batch_size*seq_len
        seq_len = text.shape[1]
        # text = [sent len, batch size]
        # text = text.permute(1, 0)
        # text = [batch size, sent len]
        embedded = self.embedding(text)  # batch_size*se1_len*emb_dim
        # embedded = [batch size, sent len, emb dim]
        embedded = embedded.unsqueeze(1)  # batch_size*1*seq_len#emb_dim
        # embedded = [batch size, 1, sent len, emb dim]
        pooled = []
        for i in range(len(self.filter_sizes)):
            conved = F.relu(self.convs[i](embedded))  # batch_size*out_channels*(seq_len-2)*1
            conved = F.max_pool2d(conved, (seq_len - self.filter_sizes[i] + 1, 1))  # batch_size*out_channels*1*1
            pooled.append(conved)

        # conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]  # batch_size*n_filters*[18,17,16]
        # # conved_n = [batch size, n_filters, sent len - filter_sizes[n] + 1]
        # pooled = [F.max_pool2d(conv,()) for conv in conved]  # 提取最显著的词
        # pooled_n = [batch size, n_filters]
        cat = self.dropout(torch.cat(pooled, dim=1)).squeeze(-1).squeeze(-1)
        # cat = [batch size, n_filters * len(filter_sizes)] [1,2,3,4,5 ngram]
        return self.fc(cat)  # batch_size*n_classes
