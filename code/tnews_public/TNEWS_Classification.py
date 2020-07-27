
import torch
import numpy as np
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import Adam
from sklearn.metrics import precision_score, recall_score, f1_score
import time

# 个人小建议：系统的包放在上面，个人的包放在下面，这样遇到word_dir出问题的时候中间可以直接修改work_dir
from utils.tokenizer import Tokenizer
from utils.get_emb import *
from models.LSTMClassifier import LSTMClassifierNet
from models.CNNClassifier import CNNClassifierNet
from models.LSTMAttentionClassifier import LSTMAttentionClassifierNet
from dataset_readers.Tnews_MultiClassCorp import *
from utils.label import *

"""超参数设定"""
# hidden_size = 1024  # 使用RNN变种LSTM单元   LSTM的hidden size
# num_layers = 1      # 循环单元/LSTM单元的层数
epoch = 5           # 迭代轮次
# num_samples = 1000  # 测试语言模型生成句子时的样本数
batch_size = 16     # 一批样本的数量
seq_length = 35     # 一个样本/序列长度
learning_rate = 0.002 # 学习率


def load_data(seq_length, label):
    print("Start to load data.")
    start_time = time.time()
    # emb: collections.OrderedDict()顺序字典
    ## key:词(str类型), value:词向量(list类型，元素为float)
    # dict_length: 字典大小
    # emb_size: 词向量的维数
    emb, dict_length, emb_size = get_emb()
    # 用所有的词(str类型)实例化一个tokenizer
    tokenizer = Tokenizer(emb.keys())
    # emb_matrix: ID与词向量的对应的矩阵
    ## ID: 每种字对应一个ID号，比如“的”1号，“是”2号以此类推
    ## 矩阵第一维的坐标就是ID号，ID号这一行的向量即对应的词向量
    emb_matrix = get_emb_matrix(emb, tokenizer, dict_length, emb_size)

    # 生成ChnSentiCorp_Clf类的实例
    ## 类的构造函数已经将数据切分成训练数据和测试数据
    data_loader = Tnews_ChnCorp_Clf(label)
    # 获取训练数据
    ## list类型，以data_example类的实例为元素
    ## data_example类包含2个属性：text，str类型；label，str类型
    train_examples = data_loader.get_train_examples()
    # 获取验证数据
    ## 同train_examples
    dev_examples = data_loader.get_dev_examples()

    def generate_dataloader(examples, tokenizer, seq_length):
        """
        生成数据加载器
        :param examples: list类型，以data_example类的实例为元素。
                        data_example类包含2个属性：text，str类型；label_id，int类型
        :param tokenizer:
        :param seq_length: 一个样本/序列长度
        :return: dataloader，迭代器类型；
        """
        features = multi_convert_example_to_feature(examples, tokenizer, seq_length)
        # ids，tensor类型(转自list类型)
        # 每个元素代表一个样本的text文本对应的ID号序列，list类型
        # 一个字对应一个ID号
        ids = torch.tensor([f.ids for f in features], dtype=torch.long)
        # labels，tensor类型(转自list类型)
        # 每个元素是一个样本对应的标签ID号
        label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)

        dataset = TensorDataset(ids, label_ids)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        return dataloader
           
    train_dataloader = generate_dataloader(train_examples, tokenizer, seq_length)
    dev_dataloader = generate_dataloader(dev_examples, tokenizer, seq_length)

    end_time = time.time()
    print("Data loading finishes. Time span: {:.2f}s".format(end_time - start_time))

    return emb_matrix, train_dataloader, dev_dataloader, tokenizer


def load_model(seq_length, label_len, emb_matrix):
    print("Start to load model.")
    start_time = time.time()
    # TODO: you can choose different model
    # model = CNNClassifierNet(seq_length, label_num, emb_matrix)
    # model = LSTMClassifierNet(seq_length, label_num, emb_matrix, bidirectional=True)
    model = LSTMAttentionClassifierNet(seq_length, label_len, emb_matrix, hidden_dims = 100, bidirectional=True)
    if torch.cuda.is_available():
        model.to(torch.device('cuda'))
    optimizer = Adam(model.parameters(), lr=learning_rate)

    end_time = time.time()
    print("Model loading finishes. Time span: {:.2f}s".format(end_time - start_time))

    return model, optimizer


def train(model, optimizer, train_dataloader, dev_dataloader, epoch=5):
    print("Start to train the model.")
    start_time = time.time()
    for i in range(epoch):
        model.train()
        total_loss = []
        for ids, label_ids in train_dataloader:
            if torch.cuda.is_available():
                ids = ids.to(torch.device('cuda'))
                label_ids = label_ids.to(torch.device('cuda'))
            optimizer.zero_grad()
            # 为模型传入字符的ID列表和标签ID列表
            # 当传入标签，模型认为此时为训练状态，则返回损失值
            loss = model(ids, label_ids)
            # loss为标量，0维
            # 使用loss.item()从标量中获取Python数字
            total_loss.append(loss.item())
            loss.backward()
            optimizer.step()
        print("epoch: %d, loss: %.6f" % (i + 1, sum(total_loss) / len(total_loss)))

        model.eval()
        total_labels = []
        total_pred = []
        for ids, label_ids in dev_dataloader:
            if torch.cuda.is_available():
                ids = ids.to(torch.device('cuda'))
            # logits即表示分类模型产生的一个预测结果，一般接着输入Softmax
            # logits (batch_size, label_num)
            # the logits vector of raw (non-normalized) predictions that a classification model generates,
            # which is ordinarily then passed to a normalization function.
            # If the model is solving a multi-class classification problem,
            # logits typically become an input to the softmax function.
            # The softmax function then generates a vector of (normalized) probabilities with one value for each possible class.
            logits = model(ids)
            # 转成numpy
            logits = logits.detach().cpu().numpy()
            # 在倒数第一个维度上求最大值的下标
            # m = [
            #       [1, 2, 3],
            #       [4, 8, 6]
            #      ]
            # logits = np.argmax(m, axis=-1)
            # print(logits)  # [2 1]
            # 即求每个batch上label概率值最大的下标
            logits = np.argmax(logits, axis=-1)
            # 将numpy转换成list类型
            logits = logits.tolist()
            # 追加入total_pred列表后
            total_pred.extend(logits)
            label_ids = label_ids.numpy().tolist()
            total_labels.extend(label_ids)
        # eval_p = precision_score(total_labels, total_pred)
        # eval_r = recall_score(total_labels, total_pred)

        # The F1 Score is the 2*((precision*recall)/(precision+recall))
        # F1 score conveys the balance between the precision and the recall.
        # 多分类问题，应设置average参数
        eval_f1 = f1_score(total_labels, total_pred, average='micro')
        print("eval_f1: %.2f%%" % (eval_f1 * 100))

    end_time = time.time()
    print("Model training finishes. Time span: {:.2f}s".format(end_time - start_time))

def tensor_to_label(logits):
    """

    :param logits: 预测结果的概率分布，(1, label_num)；因为输入仅一段话，故第一维度是1
    :return:
    """
    # detach()就是截断反向传播的梯度流
    # 将logits转化成numpy()
    # logits (batch_size, label_num)
    logits = logits.detach().cpu().numpy()
    # 选出每个预测的分布中，概率最大值的下标
    pred = np.argmax(logits, axis=-1)
    return label.id_to_desc[pred[0]]


def test(model, tokenizer, seq_length):
    """
    测试模型：判断输入字符串的文本类型
    """
    print("Start to test the model.")
    while True:
        s = input()
        if s == 'quit':
            break
        s = [data_example(s, 0)]
        s_features = multi_convert_example_to_feature(s, tokenizer, seq_length)
        ids = torch.tensor([f.ids for f in s_features], dtype=torch.long)
        with torch.no_grad():
            if torch.cuda.is_available():
                ids = ids.to(torch.device('cuda'))

            res = tensor_to_label(model(ids))
            print(res)
        print("Stop testing.")


if __name__ == '__main__':
    label = Label()
    label_num = label.cal_label_num()
    print("label_num:", label_num)
    emb_matrix, train_dataloader, dev_dataloader, tokenizer = load_data(seq_length, label)

    model, optimizer = load_model(seq_length, label_num, emb_matrix)
    train(model, optimizer, train_dataloader, dev_dataloader, epoch=epoch)

    # 使用模型
    test(model, tokenizer, seq_length)
