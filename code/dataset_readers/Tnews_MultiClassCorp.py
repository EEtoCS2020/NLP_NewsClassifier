from utils.data_utils import *
from sklearn.model_selection import train_test_split
import json
from utils.label import *
# import os
# os.path.abspath(os.path.join(os.getcwd(), '..'))

class Tnews_ChnCorp_Clf:

    def __init__(self, label: Label):
        # 读原始数据
        # self.examples = []
        self.train_set, self.dev_set, self.test_set = [], [], []
        # 加载训练集数据
        #train_filename_json = '/content/drive/My Drive/pkuss-nlp-TNEWS-Multiclass/tnews_public/train.json'
        train_filename_json = './tnews_public/train.json'
        json_file = open(train_filename_json, 'r', encoding='utf-8')
        for line in json_file.readlines():
            pop_dict = json.loads(line)
            # 获取到"label_desc"的内容
            # 通过Label类的方法desc_to_id()获取label的id号
            label_id = label.desc_to_id(pop_dict["label_desc"])
            sentence = pop_dict["sentence"] + pop_dict["keywords"]
            self.train_set.append(data_example(sentence, label_id))

        # 加载验证集数据
        #dev_filename_json = '/content/drive/My Drive/pkuss-nlp-TNEWS-Multiclass/tnews_public/dev.json'
            dev_filename_json = './tnews_public/dev.json'
        json_file = open(dev_filename_json, 'r', encoding='utf-8')
        for line in json_file.readlines():
            pop_dict = json.loads(line)
            label_id = label.desc_to_id(pop_dict["label_desc"])
            sentence = pop_dict["sentence"] + pop_dict["keywords"]
            self.dev_set.append(data_example(sentence, label_id))

        # 加载测试集数据
        #test_filename_json = '/content/drive/My Drive/pkuss-nlp-TNEWS-Multiclass/tnews_public/test.json'
            test_filename_json = './tnews_public/test.json'
        json_file = open(test_filename_json, 'r', encoding='utf-8')
        for line in json_file.readlines():
            pop_dict = json.loads(line)
            sentence = pop_dict["sentence"] + pop_dict["keywords"]
            self.test_set.append(data_example(sentence, None))

    def get_train_examples(self):
        return self.train_set

    def get_dev_examples(self):
        return self.dev_set

    def get_test_examples(self):
        return self.test_set
