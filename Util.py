import os
import re
from time import time
import random
import Constants
import sys
import torch
import json
import numpy as np

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"


def load_weights(word2index, path):
    weights = [word for word, idx in word2index.items()]
    index2word = {idx: word for word, idx in word2index.items()}

    print("加载预训练词向量", os.path.abspath(path))
    f = open(path, "r", encoding="utf-8")
    embed_dim = 300
    for line in f:
        tokens = line.strip().split(" ")
        if len(tokens) < 300:
            print("预训练词向量规模", line)
            continue
        if tokens[0] not in word2index:
            continue
        for i in range(1, 1 + embed_dim):
            tokens[i] = float(tokens[i])
        weights[word2index[tokens[0]]] = tokens[1:1 + embed_dim]

    strange_words = []
    for i in range(len(weights)):
        if len(weights[i]) == embed_dim:
            continue
        else:
            weights[i] = np.random.randn(embed_dim).tolist()
            strange_words.append(index2word[i])

    print(len(strange_words), "个生词随机初始化", " ".join(strange_words))
    return torch.Tensor(weights)


def unicode_type(unicode):  # 判断字符类型
    # https://gist.github.com/shingchi/64c04e0dd2cbbfbc1350
    if ord(unicode) <= 0x007f:  # ascii
        if ord(unicode) >= 0x0041 and ord(unicode) <= 0x005a:
            return "latin"
        if ord(unicode) >= 0x0061 and ord(unicode) <= 0x007a:
            return "latin"
        return "ascii_symble"
    if ord(unicode) >= 0x4E00 and ord(unicode) <= 0x9fff:
        return "han"  # 标准CJK文字
    if ord(unicode) >= 0xFF00 and ord(unicode) <= 0xFFEF:
        return "han_symble"  # 全角ASCII、全角中英文标点、半宽片假名、半宽平假名、半宽韩文字母：FF00-FFEF
    if ord(unicode) >= 0x3000 and ord(unicode) <= 0x303F:
        return "han_symble"  # CJK标点符号：3000-303F
    return "other"


def split_lans(line):
    last_latin = None
    grams = []
    for gram in line:
        if unicode_type(gram) == "latin":
            if last_latin == None or last_latin == False:
                grams.append(gram)
            else:
                grams[-1] += gram
            last_latin = True
        else:
            grams.append(gram)
            last_latin = False
    return grams


def count_word(counter, word, n=1):  # 统计词频  累加n
    if word not in counter:
        counter[word] = n
    else:
        counter[word] += n


def sort_counter(counter, reverse=True):  # 词频降序
    items = sorted(counter.items(), key=lambda kv: kv[1], reverse=reverse)
    counter = dict(items)
    return counter


def counter2frequency(counter):
    sum = 0
    for word, num in counter.items():
        sum += num
    frequency = {}
    for word, num in counter.items():
        frequency[word] = num / sum
    return frequency


def counter2dict(counter, word2index=Constants.Default_Dict, min_freq=2, max_token=10000):  # 生成字典
    ignored_word_count = 0
    for word, count in counter.items():
        if len(word2index) >= max_token:
            print("词典已满")
            break
        if word not in word2index:
            if count >= min_freq:
                word2index[word] = len(word2index)
            else:
                ignored_word_count += 1
    print('[Info] 频繁字典大小 = {},'.format(len(word2index)), '最低频数 = {}'.format(min_freq))
    print("[Info] 忽略罕词数 = {}".format(ignored_word_count))
    return word2index


def get_index2word(word2index):
    index2word = []
    for word, count in word2index.items():
        index2word.append(word)
    return index2word


def sentence2indices(line, word2index, max_len=20, padding_index=Constants.PAD, unk=Constants.UNK, began=None,
                     end=None):
    result = [word2index.get(word, unk) for word in line]
    result = result[:max_len]
    if began is not None:
        result.insert(0, began)
    if end is not None:
        result.append(end)
    if max_len is not None and len(result) < max_len:
        result += [padding_index] * (max_len - len(result))

    assert len(result) == max_len

    return result


def indices2sentence(index2word, indices):
    sentence = "".join(index2word[index] for index in indices)
    return sentence


def split_train(x, rate=0.90, shuffle=True):
    if shuffle:
        random.shuffle(x)
    index = int(len(x) * rate)
    train = x[:index]
    test = x[index:]
    index = int(len(test) * 0.9)
    valid = test[:index]
    test = test[index:]
    return train, valid, test


def write_splits(x, dir="data", shuffle=True):
    if shuffle:
        random.shuffle(x)
    left = int(len(x) * 0.9)
    right = left + int(0.9 * (len(x) - left))

    with open(dir + "/train.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(x[:left]))
    with open(dir + "/valid.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(x[left:right]))
    with open(dir + "/test.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(x[right:]))
    print("训练集、验证集、测试集已写入", dir, "目录下")


def count_doc(doc, counter={}):
    # for index, line in enumerate(iter(lambda: read_start_of_line(reade_file), '')):
    for line in doc:
        words = split_lans(line)
        for word in words:
            if word:
                count_word(counter, word)
    return sort_counter(counter)


def merge_counter(counter1, counter2):
    if len(counter1) > 0:
        for word, num in counter1.items():
            count_word(counter2, word, num)
    return sort_counter(counter2)


def make_batches(list, batch_size, vocab, max_len=20, padding_index=Constants.PAD, shuffle=True, length_group=True):
    if length_group:
        list.sort(key=lambda x: len(x[0]))

    for i in range(0, len(list), batch_size):
        batch = list[i:i + batch_size]
        if shuffle:
            random.shuffle(batch)
        x, y = [], []
        for j in range(len(batch)):
            batch[j] = batch[j][:max_len]
            x.append(sentence2indices(line=batch[j], word2index=vocab, max_len=20, padding_index=Constants.PAD))
            y.append(batch[j][1])
        yield torch.LongTensor(x), torch.LongTensor(y)


def shot_batch(dict_data, n_class, n_support, n_batch, max_len, word2index=None):  # [y][x,x,x,]
    classes = random.sample([x for x in range(len(dict_data))], n_class)
    # 按类选取，标签映射至range(0,n_class)
    labels = {}  # 照此顺序，类内打乱
    for i in range(len(classes)):
        labels[classes[i]] = i

    support_x, support_y = [], []
    batch_x, batch_y = [], []

    batch = {}
    for c in classes:
        examples = dict_data[c]
        # while n_support + n_batch>len(examples):
        #     examples+=examples
        examples = random.sample(examples, n_support + n_batch)
        for i in range(len(examples)):
            examples[i] = sentence2indices(line=examples[i], word2index=word2index, max_len=max_len,
                                           padding_index=Constants.PAD)

        for example in examples[0:n_support]:  # sample_labels:[0,1,2,3,4]
            support_y.append(labels[c])
            support_x.append(example)

        batch[labels[c]] = examples[n_support:n_support + n_batch]

    for i in range(n_batch):  # batch_labels:[0,1,2,3,4]~19
        for c in range(n_class):
            batch_y.append(c)
            batch_x.append(batch[c][i])

    # samples, sample_labels, batches, batch_labels = [], [], [], []

    samples, sample_labels, batches, batch_labels = torch.LongTensor(support_x), torch.LongTensor(
        support_y), torch.LongTensor(
        batch_x), torch.LongTensor(batch_y)
    return samples.to(device), sample_labels.to(device), batches.to(device), batch_labels.to(device), labels


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def json_dict():
    path = "data/chitchat_data.10000.txt"

    doc = open(path, "r", encoding="utf-8").read().splitlines()
    counter = count_doc(doc)
    with open("data/counter.json", "w", encoding="utf-8") as f:
        json.dump(counter, f, ensure_ascii=False)
    dict = counter2dict(counter)
    print(dict)

    labels = {}
    for line in doc:
        line = line.split("\t")[0]
        if line not in labels:
            labels[line] = len(labels)
    print(len(labels))


if __name__ == "__main__":
    t0 = time()
    json_dict()
    print("耗时", time() - t0)
