from __future__ import print_function
import torch
from torch.autograd import Variable
import torch.nn as nn
import os
import Constants
import torch
import torch.nn as nn
import torch.nn.functional as F
import StructuredSelfAttention
from yelp_reader import *
from yelp_util import *
# from yelp_models import *
from StructuredSelfAttention import *
from yelp_train import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main():
    config = {
        "emb_dim": 64,
        "batch_size": 5,  # 尽量整除
        "lstm_hid_dim": 50,
        "d_a": 100,
        "r": 10,
        "max_len": 28, #80
        "n_classes": 5,
        "num_layers": 1,
        "dropout": 0.1,
        "type": 1,
        "emb_dim": 128,
        "use_pretrained_embeddings": False,
        "embeddings": None,
        "epochs": 200,
    }
    reader = Reader(config)
    data_dir = "../data/fewshot"
    # path = os.path.abspath(os.path.join(os.getcwd(), "../data/"))
    path = data_dir + "/yelp_review_subset.json"
    x, y = reader.read(path)
    config["vocab_size"] = len(reader.word2index)
    train_loader, valid_loader, test_loader = get_dataloader(x, y, batch_size=config["batch_size"])

    encoder_model = StructuredSelfAttention(config).to(device)
    print("encoder_model", encoder_model)
    avg_sentence_embeddings = encode(encoder_model, train_loader, config)
    print("encoder训练完毕！")


def encode(model, train_loader, config):
    loss = torch.nn.NLLLoss()
    optimizer = torch.optim.RMSprop(model.parameters())
    losses, accuracy, avg_sentence_embeddings = train(model, train_loader, loss, optimizer, epochs=config["epochs"],
                                                      use_regularization=True, C=1.0, clip=True)
    return avg_sentence_embeddings


def train0(config):
    img_shape = (1, 28, 28)
    matching_net_trial = MatchingNet(img_shape, dropout_probality=0.1, use_fce=False)
    print("Model Summary")
    print(matching_net_trial)
    epochs = 10

    support_images = torch.rand(32, 20, *img_shape)
    target_image = torch.rand(32, *img_shape)
    support_labels = torch.LongTensor(32, 20, 1) % 20
    target_labels = torch.LongTensor(32) % 20

    matching_net_trial.to(device)
    support_images = support_images.to(device)
    support_labels = support_labels.to(device)
    target_image = target_image.to(device)
    target_labels = target_labels.to(device)
    optimizer = torch.optim.Adam(matching_net_trial.parameters(), lr=0.001)
    for epoch in range(epochs):
        logits, predictions = matching_net_trial(support_images, support_labels, target_image)
        loss = F.cross_entropy(logits, target_labels)
        print(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


import sys

if __name__ == "__main__":
    main()
