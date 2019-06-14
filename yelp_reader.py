import json
import torch
from torch.autograd import Variable
import Constants
import os
import torch.utils.data as Data
import Constants
'''
 不同语言、汉字之间、标点文字之间、数字之间加空格
'''

class Reader():
    def __init__(self,config):
        self.max_len=config["max_len"]
        self.word2index={Constants.PAD_WORD:0}
        self.index2word=[]

    def lan_type(self,unicode):
        if ord(unicode) <=0x007f:
            if ord(unicode) >= 0x0041 and ord(unicode) <= 0x005a:
                return "latin"
            if ord(unicode) >= 0x0061 and ord(unicode) <= 0x007a:
                return "latin"
            return "symble"
        if ord(unicode) >= 0x4E00 and ord(unicode) <= 0x9fff:
            return "han"
        return False

    def gram_break(self,line):
        # print("之前",line)
        # if "%" in line:
        #     print(line)
        gram=[]
        for word in line:
            if self.lan_type(word) in ["symble","han"]:
                word=" "+word+" "
            gram.append(word)
        line2="".join(gram)
        line2=" ".join(line2.split())
        # if len(line2)>self.max_len:
        #     return line2[:self.max_len]
        # print("之后",line2)
        return line2

    def add_word(self,word):
        if word not in self.word2index:
            self.word2index[word]=len(self.word2index)
            self.index2word.append(word)

    def sentence2indices(self,line):
        for word in list(line):
            self.add_word(word)
        result= [self.word2index[word] for word in list(line)]
        if len(result)>=self.max_len:
            result= result[:self.max_len]
        else:
            b=[Constants.PAD]*(self.max_len-len(result))
            result+=b
        assert len(result)==self.max_len
        return result

    def indices2sentence(self, indices):
        sentence = "".join(self.index2word[index] for index in indices)
        return sentence

    def digitize(self, origin):
        batch=[]
        for line in origin:
            # print(line)
            batch.append(self.sentence2indices(self.word2index,line))
        # print(batch[0])
        # print( Variable(torch.LongTensor(batch)).shape)
        return Variable(torch.LongTensor(batch))

    #data_url = 'http://apache-mxnet.s3-accelerate.dualstack.amazonaws.com/gluon/dataset/yelp_review_subset-167bb781.zip'
    def read(self,path,train_rate=0.8):
        print("当前工作目录", os.path.abspath(path))

        with open(path, 'r', encoding='utf-8') as fr:
            data = json.load(fr)

        texts=data["texts"]
        labels=data["labels"]
        # dataset = [[text, int(label)-1] for text, label in zip(data['texts'], data['labels'])]
        print(path + "读取完毕，共有", len(texts), "选取100")
        texts=texts[:100]
        labels=labels[:100]
        print("前四条样例如下")
        print(texts[:5])

        for i in range(len(texts)):
            labels[i]=int(labels[i])-1
            if labels[i]>=4:
                labels[i]=4
            elif labels[i]<=0:
                labels[i]=0
            texts[i]=self.gram_break(texts[i])
            for word in texts[i].split():
                self.add_word(word)

            texts[i]=self.sentence2indices(texts[i])

        print("字典生成，共",len(self.word2index))
        print(self.word2index)
        texts=torch.LongTensor(texts)
        labels=torch.LongTensor(labels)
        # texts=Variable(torch.LongTensor([texts]))
        # labels=Variable(torch.LongTensor([labels]))
        assert len(texts)==len(labels)
        return texts,labels

def get_dataloader(x,y,batch_size=10):
    assert len(x)==len(y)
    index=int(len(x)*0.8)
    train_x, train_y=x[:index],y[:index]
    test_x, test_y=x[index:],y[index:]
    index=int(len(test_x)/2)
    valid_x, valid_y=test_x[:index],test_y[index:]
    test_x, test_y=test_x[:index],test_y[index:]

    train=Data.TensorDataset(train_x, train_y)
    train_loader=Data.DataLoader(dataset=train,batch_size=batch_size,num_workers=4,shuffle=True)
    test = Data.TensorDataset(test_x, test_y)
    test_loader = Data.DataLoader(dataset=test, batch_size=batch_size,num_workers=4, shuffle=True)
    valid_loader=Data.DataLoader(dataset= Data.TensorDataset(valid_x, valid_y),batch_size=batch_size,num_workers=4,shuffle=True)


    return train_loader,valid_loader,test_loader



