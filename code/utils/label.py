import json
import collections
import sys
# sys.path.append('/content/drive/My Drive/pkuss-nlp-TNEWS-Multiclass')


class Label:
    """
    labels.json文件中的label的原ID号不连续，在构造函数中重新为label编ID
    ID从0到14
    """

    def __init__(self):
        self.list_id_to_desc = []
        self.dict_desc_to_id = collections.OrderedDict()
        # labels_filename_json = '/content/drive/My Drive/pkuss-nlp-TNEWS-Multiclass/tnews_public/labels.json'
        labels_filename_json = './tnews_public/labels.json'
        json_file = open(labels_filename_json, 'r', encoding='utf-8')
        label_id = 0
        for line in json_file.readlines():
            # 将每一行的字符串line转换成字典
            pop_dict = json.loads(line)
            # 获取标签的描述信息
            label_desc = pop_dict["label_desc"]
            # id至标签描述
            # 按照顺序追加到列表尾部
            # ID从0到14
            self.list_id_to_desc.extend([label_desc])
            self.dict_desc_to_id[label_desc] = label_id
            label_id += 1

    def cal_label_num(self) -> int:
        """
        计算标签数量
        :return:
        """
        return len(self.list_id_to_desc)

    def id_to_desc(self, label_id):
        return self.list_id_to_desc[label_id]

    def desc_to_id(self, label_desc):
        return self.dict_desc_to_id.get(label_desc, None)

    def get_labels(self):
        return self.list_id_to_desc
