import torch.nn as nn
from transformers import BertModel

class ModelManager(nn.Module):
    def __init__(self):
        super(ModelManager, self).__init__()
        self.encoder = BertEncoder()
        self.decoder = linear_classification(768, 3)

    def forward(self, input_data, attention_mask):
        hidden = self.encoder(input_data, attention_mask)
        res = self.decoder(hidden)
        return res


class BertEncoder(nn.Module):
    def __init__(self):
        super(BertEncoder, self).__init__()
        self.bert_encoder = BertModel.from_pretrained("bert-base-chinese")
        self.dropout = nn.Dropout(0.5)

    def forward(self, input_data, attention_mask):
        cls_hidden = self.bert_encoder(input_data, attention_mask=attention_mask)[0]
        cls_hidden = cls_hidden[:, 0, :]
        res = self.dropout(cls_hidden)
        return res


class linear_classification(nn.Module):
    def __init__(self, n_hidden, n_class):
        super(linear_classification, self).__init__()
        self.linear = nn.Linear(n_hidden, n_class)

    def forward(self, input_data):
        res = self.linear(input_data)
        return res
