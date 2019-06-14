import torch
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as data_utils

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import Constants


class StructuredSelfAttention(torch.nn.Module):
    """
    The class is an implementation of the paper A Structured Self-Attentive Sentence Embedding including regularization
    and without pruning. Slight modifications have been done for speedup
    https://github.com/kaushalshetty/Structured-Self-Attention
    """

    def __init__(self, config):
        # batch_size = config["batch_size"]
        lstm_hid_dim = config["lstm_hid_dim"]
        d_a = config["d_a"]
        r = config["r"]
        max_len = config["max_len"]
        emb_dim = config["emb_dim"]
        vocab_size = config["vocab_size"]
        use_pretrained_embeddings = config["use_pretrained_embeddings"]
        embeddings = config["embeddings"]
        type = 1
        n_classes = config["n_classes"]
        num_layers = config["num_layers"]
        """
        Initializes parameters suggested in paper

        Args:
            batch_size  : {int} batch_size used for training
            lstm_hid_dim: {int} hidden dimension for lstm
            d_a         : {int} hidden dimension for the dense layer
            r           : {int} attention-hops or attention heads
            max_len     : {int} number of lstm timesteps
            emb_dim     : {int} embeddings dimension
            vocab_size  : {int} size of the vocabulary
            use_pretrained_embeddings: {bool} use or train your own embeddings
            embeddings  : {torch.FloatTensor} loaded pretrained embeddings
            type        : [0,1] 0-->binary_classification 1-->multiclass classification
            n_classes   : {int} number of classes

        Returns:
            self

        Raises:
            Exception
        """
        super(StructuredSelfAttention, self).__init__()
        self.num_layers = num_layers
        self.embeddings = self._load_embeddings(use_pretrained_embeddings, embeddings, vocab_size, emb_dim)
        self.embeddings.requires_grad = False
        self.lstm = torch.nn.LSTM(emb_dim, lstm_hid_dim, num_layers=self.num_layers, batch_first=True)
        # torch.nn.init.xavier_normal(self.lstm)
        self.linear_first = torch.nn.Linear(lstm_hid_dim, d_a)
        torch.nn.init.xavier_uniform(self.linear_first.weight)
        # self.linear_first.bias.data.fill_(0)
        self.linear_second = torch.nn.Linear(d_a, r)
        torch.nn.init.xavier_uniform(self.linear_second.weight)
        # self.linear_second.bias.data.fill_(0)
        self.n_classes = n_classes
        self.linear_final = torch.nn.Linear(lstm_hid_dim, self.n_classes)
        torch.nn.init.xavier_uniform(self.linear_final.weight)

        self.batch_size = 0
        self.max_len = max_len
        self.lstm_hid_dim = lstm_hid_dim
        self.hidden_state = None
        self.r = r
        self.type = type
        self.dropout = torch.nn.Dropout(0.1)

    def _load_embeddings(self, use_pretrained_embeddings, embeddings, vocab_size, emb_dim):
        """Load the embeddings based on flag"""

        if use_pretrained_embeddings is True and embeddings is None:
            raise Exception("Send a pretrained word embedding as an argument")

        if not use_pretrained_embeddings and vocab_size is None:
            raise Exception("Vocab size cannot be empty")

        if not use_pretrained_embeddings:
            word_embeddings = torch.nn.Embedding(vocab_size, emb_dim, padding_idx=Constants.PAD)

        elif use_pretrained_embeddings:
            word_embeddings = torch.nn.Embedding(embeddings.size(0), embeddings.size(1))
            word_embeddings.weight = torch.nn.Parameter(embeddings)
            emb_dim = embeddings.size(1)

        return word_embeddings  # weights=wocab_size*emb_dim

    def softmax(self, input, axis=1):
        """
        Softmax applied to axis=n
        Args:
           input: {Tensor,Variable} input on which softmax is to be applied
           axis : {int} axis on which softmax is to be applied
        Returns:
            softmaxed tensors
        """
        input_size = input.size()
        trans_input = input.transpose(axis, len(input_size) - 1)
        trans_size = trans_input.size()
        input_2d = trans_input.contiguous().view(-1, trans_size[-1])
        soft_max_2d = F.softmax(input_2d)
        soft_max_nd = soft_max_2d.view(*trans_size)
        return soft_max_nd.transpose(axis, len(input_size) - 1)

    def init_hidden(self, batch_size):
        # return (Variable(torch.zeros(self.num_layers, batch_size, self.lstm_hid_dim)).to(device),
        #         Variable(torch.zeros(self.num_layers, batch_size, self.lstm_hid_dim)).to(device))
        a = Variable(torch.randn(self.num_layers, batch_size, self.lstm_hid_dim)).to(device)
        b = Variable(torch.randn(self.num_layers, batch_size, self.lstm_hid_dim)).to(device)
        self.batch_size = batch_size
        self.hidden_state = (a, b)

    def forward(self, x):  # batch_size*max_len
        x = x.to(device)
        # print("StructuredSelfAttention输入形状",x.shape)
        embeddings = self.embeddings(x)  # batch_size*max_len*emb_dim
        if self.batch_size != x.shape[0]:
            self.init_hidden(x.shape[0])

        outputs, self.hidden_state = self.lstm(embeddings, self.hidden_state)  # batch_size*max_len*emb_dim
        # outputs batch_size*max_len*lstm_hid_dim
        x = F.tanh(self.linear_first(self.dropout(outputs)))  # batch_size*max_len*d_a
        x = self.linear_second(self.dropout(x))  # batch_size*max_len*r
        x = self.softmax(x, 1)
        attention = x.transpose(1, 2)  # batch_size*r*max_len
        sentence_embeddings = attention @ outputs  # batch_size*r*lstm_hid_dim
        avg_sentence_embeddings = torch.sum(sentence_embeddings, 1) / self.r  # batch_size*lstm_hid_dim
        #
        # if not bool(self.type):
        #     output = F.sigmoid(self.linear_final(avg_sentence_embeddings))
        #     return output, attention, avg_sentence_embeddings
        # else:  # 多分类
        #     output = F.log_softmax(self.linear_final(avg_sentence_embeddings))  #batch_size*n_classes
        #     return output, attention, avg_sentence_embeddings
        # 只提取特征
        # return F.log_softmax(avg_sentence_embeddings)
        return avg_sentence_embeddings


# Regularization
def l2_matrix_norm(self, m):
    """
    Frobenius norm calculation
    Args:
       m: {Variable} ||AAT - I||
    Returns:
        regularized value
    """
    return torch.sum(torch.sum(torch.sum(m ** 2, 1), 1) ** 0.5).to(device)
