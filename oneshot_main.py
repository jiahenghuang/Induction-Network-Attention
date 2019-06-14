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
from time import time
from torch.optim.lr_scheduler import StepLR
from torch.autograd import Variable
import numpy as np
from StructuredSelfAttention import *
from trainer import *

# Hyper Parameters
FEATURE_DIM = 50
RELATION_DIM = 8
CLASS_NUM = 5
SAMPLE_NUM_PER_CLASS = 1
BATCH_NUM_PER_CLASS = 19
EPISODE = 1000000
TEST_EPISODE = 10
LEARNING_RATE = 0.01
HIDDEN_UNIT = 10


def main():
    # path = "data/chitchat_data.10000.txt"
    path = "data/oneshot.txt"
    doc = open(path, "r", encoding="utf-8").read().splitlines()
    counter = count_doc(doc)
    # with open("data/counter.json", "w", encoding="utf-8") as f:
    #     json.dump(counter, f, ensure_ascii=False)
    word2index = counter2dict(counter=counter, min_freq=2)
    print(word2index)

    # weights = load_weights(word2index=word2index, path="../data/wordvec/merge_sgns_bigram_char300.txt")
    weights = None
    config = {
        "emb_dim": 64,
        "lstm_hid_dim": 50,
        "d_a": 100,
        "r": 10,
        "max_len": 28,
        "n_classes": 5,
        "num_layers": 1,
        "dropout": 0.1,
        "type": 1,
        "emb_dim": 128,
        "use_pretrained_embeddings": False,
        "embeddings": weights,
        "epochs": 200,
        "vocab_size": len(word2index)
    }
    encoder_model = StructuredSelfAttention(config).to(device)

    # dir="data"
    # print('[Info] 保存训练数据到', os.path.abspath("data/word2index.data"))
    # torch.save(word2index, path)
    # torch.save(weights, path)
    # print('[Info] Finished.')
    # # load

    labels = {}
    for line in doc:
        line = line.split("\t")[0]
        if line not in labels:
            labels[line] = len(labels)
    print("标签类别数量", len(labels))

    # 使用列表不好抽取
    dict_data = {}
    for line in doc:
        y, x = line.split("\t")
        y = labels[y]
        if y not in dict_data:
            dict_data[y] = []
        dict_data[y].append(x)

    # feature_encoder = CNNEncoder(vocab_size=weights.shape[0], embed_dim=weights.shape[1], weights=weights.to(device))
    relation_network = RelationNetwork(2*FEATURE_DIM, RELATION_DIM)
    # feature_encoder.apply(weights_init).to(device)
    feature_encoder=encoder_model
    relation_network.apply(weights_init).to(device)

    feature_encoder_optim = torch.optim.Adam(feature_encoder.parameters(), lr=LEARNING_RATE)
    feature_encoder_scheduler = StepLR(feature_encoder_optim, step_size=100, gamma=0.9)
    relation_network_optim = torch.optim.Adam(relation_network.parameters(), lr=LEARNING_RATE)
    relation_network_scheduler = StepLR(relation_network_optim, step_size=100, gamma=0.9)

    if os.path.exists(str(
            "./models/omniglot_feature_encoder_" + str(CLASS_NUM) + "way_" + str(SAMPLE_NUM_PER_CLASS) + "shot.pkl")):
        feature_encoder.load_state_dict(torch.load(str(
            "./models/omniglot_feature_encoder_" + str(CLASS_NUM) + "way_" + str(SAMPLE_NUM_PER_CLASS) + "shot.pkl")))
        print("load feature encoder success")
    if os.path.exists(str(
            "./models/omniglot_relation_network_" + str(CLASS_NUM) + "way_" + str(SAMPLE_NUM_PER_CLASS) + "shot.pkl")):
        relation_network.load_state_dict(torch.load(str(
            "./models/omniglot_relation_network_" + str(CLASS_NUM) + "way_" + str(SAMPLE_NUM_PER_CLASS) + "shot.pkl")))
        print("load relation network success")

    print("开始训练")
    t0 = time()
    last_accuracy = 0.0

    for episode in range(EPISODE):
        feature_encoder.train()
        relation_network.train()
        feature_encoder_scheduler.step(episode)
        relation_network_scheduler.step(episode)

        loss = train(feature_encoder, relation_network, dict_data, word2index)

        feature_encoder_optim.step()
        relation_network_optim.step()

        if (episode + 1) % 10 == 0:
            print("episode:", episode + 1, "loss", loss, "耗时", time() - t0)
            t0 = time()

        if (episode + 1) % 100 == 0:
            test_accuracy = valid(feature_encoder, relation_network, dict_data, word2index)

            if test_accuracy >= last_accuracy:
                # save networks
                torch.save(feature_encoder.state_dict(), str(
                    "./models/omniglot_feature_encoder_" + str(CLASS_NUM) + "way_" + str(
                        SAMPLE_NUM_PER_CLASS) + "shot.pkl"))
                torch.save(relation_network.state_dict(), str(
                    "./models/omniglot_relation_network_" + str(CLASS_NUM) + "way_" + str(
                        SAMPLE_NUM_PER_CLASS) + "shot.pkl"))

                print("save networks for episode:", episode)

                last_accuracy = test_accuracy

    print("完成")


if __name__ == "__main__":
    t0 = time()
    main()
    print("耗时", time() - t0)
