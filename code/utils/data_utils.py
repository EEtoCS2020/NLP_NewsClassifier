# 原始数据和标签
class data_example:
    def __init__(self, text: str, label_id: int):
        """

        :param text: 文本内容
        :param label_id: 文本对应的标签id
        """
        self.text = text
        self.label_id = label_id


# 处理完毕的数据和标签
class data_feature:
    def __init__(self, ids: list, label_id: int):
        """

        :param ids: 字符串文本中每个字符对应的ID号
        :param label_id: 标签对应的ID号
        """
        self.ids = ids
        self.label_id = label_id



def bi_convert_example_to_feature(examples, tokenizer, seq_length):
    """
    (二分类)将原始转换成数据特征id+标签id
    :param examples: list类型，以data_example类的实例为元素。
                    data_example类包含2个属性：text，str类型；label，str类型
    :param tokenizer:
    :param seq_length: 一个样本的序列长度
    :return: features，list类型；每个元素是一个data_feature实例，包含两个属性；
                        ids，list类型，元素为ID号，examples.text(str类型)每个字符对应的ID
                        label_id，int类型，positive：1，negative：0
    """
    features = []
    for i in examples:
        # 使用tokenizer将字符串转换为ID号列表
        ## ids为list类型，元素为int类型，每个int对应字符串的一个字符
        ids = tokenizer.tokens_to_ids(i.text)
        # 我们规定了最大长度，超过了就切断，不足就补齐（一般补unk，也就是这里的[0]，也有特殊补位符[PAD]之类的）
        if len(ids) > seq_length:
            ids = ids[0: seq_length]
        else:
            ids = ids + [0] * (seq_length - len(ids))
        # 如果这个字符串全都不能识别，那就放弃掉
        if sum(ids) == 0:
            continue
        assert len(ids) == seq_length
        # 处理标签，正面为1，负面为0
        if i.label == 'positive':
            label_ids = 1
        else:
            label_ids = 0
        features.append(data_feature(ids, label_ids))
    return features


def multi_convert_example_to_feature(examples, tokenizer, seq_length):
    """
    (多分类)将原始转换成数据特征id+标签id
    :param examples: list类型，以data_example类的实例为元素。
                    data_example类包含2个属性：text，str类型；label_id，int类型
    :param tokenizer:
    :param seq_length: 一个样本的序列长度
    :return: features，list类型；每个元素是一个data_feature实例，包含两个属性；
                        ids，list类型，元素为ID号，data_example.text(str类型)每个字符对应的ID
                        label_id，int类型
    """
    features = []
    for example in examples:
        # 使用tokenizer将字符串转换为ID号列表
        ## ids为list类型，元素为int类型，每个int对应字符串的一个字符
        ids = tokenizer.tokens_to_ids(example.text)
        # 我们规定了最大长度，超过了就切断，不足就补齐（一般补unk，也就是这里的[0]，也有特殊补位符[PAD]之类的）
        if len(ids) > seq_length:
            ids = ids[0: seq_length]
        else:
            ids = ids + [0] * (seq_length - len(ids))
        # 如果这个字符串全都不能识别，那就放弃掉
        if sum(ids) == 0:
            continue
        assert len(ids) == seq_length
        features.append(data_feature(ids, example.label_id))
    return features

