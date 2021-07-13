from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
from transformers import BertTokenizer


class myDataset(Dataset):
    # x: 输入的句子序列 y：情感分类结果 mask：mask矩阵
    def __init__(self, x, y, mask):
        super(myDataset, self).__init__()
        self.sample_num = x.shape[0]
        self.x = x
        self.y = y
        self.mask = mask

    def __getitem__(self, index):
        return self.x[index], self.y[index], self.mask[index]

    def __len__(self):
        return self.sample_num

def DataProcess():
    filename = 'data/train_data.json'
    content_train, label_train = make_data_json(filename)

    filename = 'data/eval_data.json'
    content_dev, label_dev = make_data_json(filename)

    filename = 'data/test_data.json'
    content_test, label_test = make_data_json(filename)

    tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")

    train_tokenized = list(map((lambda x: tokenizer.encode(x, add_special_tokens=True,max_length=512,truncation=True)), content_train))
    dev_tokenized = list(map((lambda x: tokenizer.encode(x, add_special_tokens=True,max_length=512,truncation=True)), content_dev))
    test_tokenized = list(map((lambda x: tokenizer.encode(x, add_special_tokens=True,max_length=512,truncation=True)), content_test))

    train_padded = pad(train_tokenized)
    dev_padded = pad(dev_tokenized)
    test_padded = pad(test_tokenized)

    train_attention_mask = np.where(train_padded != 0, 1, 0)
    dev_attention_mask = np.where(dev_padded != 0, 1, 0)
    test_attention_mask = np.where(test_tokenized != 0, 1, 0)

    train_loader = DataLoader(dataset=myDataset(train_padded, label_train, train_attention_mask), batch_size=16,
                              shuffle=True)
    dev_loader = DataLoader(dataset=myDataset(dev_padded, label_dev, dev_attention_mask), batch_size=16, shuffle=True)
    test_loader = DataLoader(dataset=myDataset(test_padded, label_test, test_attention_mask), batch_size=16)
    print("Data processing done!")
    return train_loader, dev_loader, test_loader


def make_data_json(filename):
    with open(filename, 'r') as f:
        data_all = json.load(f)
        content = []
        label = []
        if filename != 'data/test_data.json':
            for data in data_all:
                content.append(data['content'])
                if data['label'] == 'positive':
                    label.append(1)
                elif data['label'] == 'neutral':
                    label.append(0)
                elif data['label'] == 'negative':
                    label.append(2)
                else:
                    label.append('null')
        else:
            for _ in data_all:
                label.append(0)
    return content, np.array(label)


def pad(tokenized):
    max_len = 0
    for i in tokenized:
        if len(i) > max_len:
            max_len = len(i)

    max_len = min(max_len, 512)

    padded = []
    for i in tokenized:
        if len(i) < max_len:
            padded.append(i + [0] * (max_len - len(i)))
        else:
            padded.append(i[0: max_len])
    return np.array(padded)

if __name__  == '__main__':
    DataProcess()