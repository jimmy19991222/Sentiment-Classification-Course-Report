import torch
import csv
from Data_Loader import *
from model import ModelManager

if __name__ == '__main__':
    model = ModelManager()
    model.load_state_dict(torch.load("model_param/parameter_1.pkl",map_location='cpu'))

    test_loader = DataProcess()

    model.eval()

    f = open('predict_1.csv', 'w', encoding='utf-8')
    csv_writer = csv.writer(f)
    num = 1

    for i, data in enumerate(test_loader):
        x, _, mask = data
        x = x.long()
        if torch.cuda.is_available():
            x = x.cuda()
            mask = mask.cuda()

        output = model(x, mask)
        # output [bs, 3]
        values, index = output.max(dim=1)
        predict = index.view(-1, 1)
        for t in predict:
            csv_writer.writerow([num, t.item()])
            num += 1
            print("读取第"+str(num)+"条数据")




