import os
import re
from time import time
import random
import Constants
import sys
import torch
import json
from Util import *
from models import *
# from TEXTCNN_trainer import *
from time import time
from torch.optim.lr_scheduler import StepLR
from torch.autograd import Variable
import numpy as np
from oneshot_main import *


def train(feature_encoder, relation_network, dict_data, vocab):
    samples, sample_labels, batches, batch_labels, labels = \
        shot_batch(dict_data, n_class=CLASS_NUM, n_support=SAMPLE_NUM_PER_CLASS, n_batch=BATCH_NUM_PER_CLASS, word2index=vocab, max_len=28)

    feature_encoder.zero_grad()
    relation_network.zero_grad()
    # sample_labels=torch.unsqueeze(sample_labels,1)  [0,1,2,3,4]
    # samples = torch.unsqueeze(samples, 1)  # 5,1,28
    # batches = torch.unsqueeze(batches, 1)  # 95,1,28  [0,1,2,3,4]~*19
    # batch_labels=torch.unsqueeze(batch_labels,1) #95*1

    # calculate features
    # y_pred, att, avg_sentence_embeddings = feature_encoder(samples)

    sample_features = feature_encoder(samples.to(device))  # 5x64*5*5   5*8400 ## 5*50
    batch_features = feature_encoder(batches.to(device))  # 20x64*5*5  95*8400  ##95*50

    # calculate relations
    # each batch sample link to every samples to calculate relations
    # to form a 100x128 matrix for relation network
    # repeat(m,n) 分别在0、1维上复制m、n次

    # sample_features_ext = sample_features.unsqueeze(0).repeat(BATCH_NUM_PER_CLASS * CLASS_NUM, 1, 1, 1, 1)  # 1,5,64,5,5-> 95*5*64*5*5
    # batch_features_ext = batch_features.unsqueeze(0).repeat(SAMPLE_NUM_PER_CLASS * CLASS_NUM, 1, 1, 1, 1)  # 1,95,64,5,5-> 5*95*5485*5
    # batch_features_ext = torch.transpose(batch_features_ext, 0, 1)  # *95*5*64*5*5
    #
    # relation_pairs = torch.cat((sample_features_ext, batch_features_ext), 2).view(-1, FEATURE_DIM * 2, 5, 5)  # [95,5,128,5,5]-> 475*128*5*5
    # relations = relation_network(relation_pairs).view(-1, CLASS_NUM)  # 475*1->95*5

    sample_features_ext = sample_features.unsqueeze(0).repeat(BATCH_NUM_PER_CLASS * CLASS_NUM, 1, 1)  # 1*5*50> 95*5*50
    batch_features_ext = batch_features.unsqueeze(0).repeat(SAMPLE_NUM_PER_CLASS * CLASS_NUM, 1, 1)  # 1*95*50 5*95*50
    batch_features_ext = torch.transpose(batch_features_ext, 0, 1)  # *95*5*50

    relation_pairs = torch.cat((sample_features_ext, batch_features_ext), 2).view(-1, FEATURE_DIM * 2)  # [95,5,100]-> 475*100
    relations = relation_network(relation_pairs).view(-1, CLASS_NUM)  # 475*1->95*5

    mse = nn.MSELoss().to(device)
    # one_hot_labels = Variable(torch.zeros(BATCH_NUM_PER_CLASS*CLASS_NUM, CLASS_NUM).scatter_(1, batch_labels.view(-1,1), 1)).to(device)
    batch_labels2 = torch.LongTensor([0] * 95)

    one_hot_labels = torch.zeros(BATCH_NUM_PER_CLASS * CLASS_NUM, CLASS_NUM).scatter_(1, batch_labels.view(-1, 1), 1).to(device)  # [95,5] 标签位置置1
    one_hot_labels = Variable(one_hot_labels).to(device)
    loss = mse(relations, one_hot_labels)

    # training
    loss.backward()

    # torch.nn.utils.clip_grad_norm(feature_encoder.parameters(), 0.5)
    # torch.nn.utils.clip_grad_norm(relation_network.parameters(), 0.5)
    return loss.item()


def valid(feature_encoder, relation_network, dict_data, vocab):
    # test
    feature_encoder.eval()
    relation_network.eval()

    total_rewards = 0

    for i in range(TEST_EPISODE):  # 训练测试 集合数量不同
        sample_images, sample_labels, test_images, test_labels, labels = \
            shot_batch(dict_data, n_class=CLASS_NUM, n_support=SAMPLE_NUM_PER_CLASS, n_batch=SAMPLE_NUM_PER_CLASS, word2index=vocab, max_len=28)
        # 5*28
        # test_images = torch.unsqueeze(test_images, 1)
        # sample_images = torch.unsqueeze(sample_images, 1)  # [5，1，28]
        # test_images = torch.unsqueeze(test_images, 1)  # [5，1，28]
        # sample_images, sample_labels = sample_dataloader.__iter__().next()
        # test_images, test_labels = test_dataloader.__iter__().next()

        # calculate features
        sample_features = feature_encoder(sample_images.to(device))  # 5x28->   #5*50
        test_features = feature_encoder(test_images.to(device))  # 5x28->   #5*50

        # calculate relations
        # each batch sample link to every samples to calculate relations
        # to form a 100x128 matrix for relation network
        sample_features_ext = sample_features.unsqueeze(0).repeat(SAMPLE_NUM_PER_CLASS * CLASS_NUM, 1, 1)  # 5*5*50
        test_features_ext = test_features.unsqueeze(0).repeat(SAMPLE_NUM_PER_CLASS * CLASS_NUM, 1, 1)  # 5*5*50
        test_features_ext = torch.transpose(test_features_ext, 0, 1)  # 5*5*50

        relation_pairs = torch.cat((sample_features_ext, test_features_ext), 2).view(-1, FEATURE_DIM * 2)  # 5*5*100 ->25*100
        relations = relation_network(relation_pairs).view(-1, CLASS_NUM)  # 5*5

        _, predict_labels = torch.max(relations.data, 1)  # 5

        rewards = [1 if predict_labels[j] == test_labels[j] else 0 for j in range(CLASS_NUM)]

        total_rewards += np.sum(rewards)

        test_accuracy = total_rewards / 1.0 / CLASS_NUM / SAMPLE_NUM_PER_CLASS / TEST_EPISODE
        print("test accuracy:", test_accuracy)

        return test_accuracy


if __name__ == "__main__":
    t0 = time()
    print("耗时", time() - t0)
