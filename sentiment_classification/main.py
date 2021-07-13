import random
from Processor import *
from Data_loader import *


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)  # 为CPU中设置种子，生成随机数
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)  # 为特定GPU设置种子，生成随机数
        torch.cuda.manual_seed_all(seed)  # 为所有GPU设置种子，生成随机数


if __name__ == '__main__':
    set_seed(10)

    dataset = DataProcess()
    processor = SentimentProcessor(dataset)
    processor.train()
