import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from model import ModelManager

class SentimentProcessor(object):
    def __init__(self,dataset):
        self.model = ModelManager()
        self.model.load_state_dict(torch.load("model_param/parameter.pkl"))
        if torch.cuda.is_available():
            self.model = self.model.cuda()

        self.train_loader, self.dev_loader, self.test_loader = dataset
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=2e-5, betas=[0.99,0.9999])

    def evaluate(self, selection):
        self.model.eval()
        if selection == 0:
            data_loader = self.train_loader
        elif selection ==1:
            data_loader = self.dev_loader
        else:
            data_loader = self.test_loader

        total_num, correct_num = 0.0, 0.0
        with torch.no_grad():
            for i, data in enumerate(data_loader):
                x, y, mask = data
                x = x.long()
                if torch.cuda.is_available():
                    x = x.cuda()
                    y = y.cuda()
                    mask = mask.cuda()

                output = self.model(x, mask)
                output = F.softmax(output, dim=-1)
                values, index = output.max(dim=1)
                predict = index.view(-1)
                correct_num += (predict == y).sum().item()
                total_num += y.shape[0]
            acc = correct_num / total_num
        return acc

    def train(self):
        best_acc = 0.0

        for epoch in range(50):
            print("start to handle {:3d} epoch".format(epoch))
            self.model.train()
            epoch_time_start = time.time()
            total_loss = 0.0
            for i, data in enumerate(self.train_loader):
                x, y, mask = data
                x = x.long()
                y = y.float()

                if torch.cuda.is_available():
                    x = x.cuda()
                    y = y.cuda()
                    mask = mask.cuda()

                output = self.model(x, mask)
                loss = self.criterion(output, y.long().view(-1))

                if (loss < 10):
                    total_loss += loss.item()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            epoch_time = time.time() - epoch_time_start

            print("轮数为：{:3d}, 模型在训练集上的时间开销为 {:.6f} 秒, 总损"
                  "失为：{:.6f};\n".format(epoch, epoch_time, total_loss))

            acc = self.evaluate(1)

            print("本轮模型在dev集上的acc为" + str(acc) + "\n")

            if acc >= max(best_acc):
                best_acc = acc
                # test_acc = self.evaluate(2)
                # print("本轮模型在dev集上表现的好,在test集中的准确率为" + str(test_acc) + "\n")
                print("本轮模型在dev集上表现的好，将写入parameter.pkl" + "\n")

                torch.save(self.model.state_dict(), 'model_param/parameter.pkl')