import collections
import numpy as np
import sys
#sys.path.append('/content/drive/My Drive/pkuss-nlp-TNEWS-Multiclass')
from utils.tokenizer import Tokenizer


def get_emb():
    """
    读取切分好的一行，返回词和词向量（numpy的矩阵）
    :return: emb, OrderedDict类型；key: 字，str类型；value: 词向量，ndarray类型
    :return: dict_length
    :return: emb_size
    """
    def get_coefs(word, *arr):
        """
        处理一行的数据
        :param word: 一个字符，str类型
        :param arr: emb_size维的词向量，都是int类型
        :return: word：字，str类型
        :return: 词向量，ndarray类型
        """
        return word, np.asarray(arr, dtype='float32')

    # skim-gram词向量的路径
    # sgnsPath = '/content/drive/My Drive/pkuss-nlp-TNEWS-Multiclass/素材/sgns.wiki.char'
    sgnsPath = './素材/sgns.wiki.char'
    with open( sgnsPath, 'r', encoding='utf-8') as emb_file:
        # 文件的开头是词表长度和词嵌入维度
        dict_length, emb_size = emb_file.readline().rstrip().split()
        print('dict_length: ', dict_length)
        print('emb_size: ', emb_size)
        dict_length, emb_size = int(dict_length), int(emb_size)
        # 读取emb_file文件的每一行，结果存到顺序词典中
        # OrderedDict字典，key: 字，str类型；value: 词向量，ndarray类型
        emb = collections.OrderedDict(get_coefs(*line.rstrip().split()) for line in emb_file.readlines())
    return emb, dict_length, emb_size


def get_emb_matrix(emb: collections.OrderedDict,
                   tokenizer: Tokenizer,
                   dict_length: int,
                   emb_size: int) -> np.ndarray:
    """
    获取词嵌入矩阵
    :param emb: 词嵌入；key: 字，str类型；value: 词向量，ndarray类型
    :param tokenizer: Tokenizer中的词表vocab，为OrderedDict类型；
                    key: 字，str类型；value: 字对应的ID，int类型
    :param dict_length:
    :param emb_size:
    :return: 词嵌入矩阵；矩阵的编号就是词的ID，矩阵每一行，就是这个ID对应的词的向量
    """
    # 生成一个全0矩阵，大小为（词典长度+1，嵌入维度）
    emb_matrix = np.zeros((1 + dict_length, emb_size), dtype='float32')


    for word, id in tokenizer.vocab.items():
        emb_vector = emb.get(word)
        if emb_vector is not None:
            # 将编号为id的词的词向量放在id行上
            emb_matrix[id] = emb_vector
    return emb_matrix
