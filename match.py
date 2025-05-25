import os
import json
import numpy as np
from sklearn.decomposition import PCA


def load_vectors(vector_dir):
    vectors = {}
    for file in os.listdir(vector_dir):
        if file.endswith(".json"):
            with open(f'{vector_dir}/{file}', 'r', encoding='utf-8') as f:
                vectors[file] = json.load(f)
    return vectors


def adjust_vector_dim(vector, target_dim):
    """使用resize直接调整向量维度"""
    return np.resize(vector, target_dim)



def cos_sim(a, b):
    print(f"向量 A 维度: {len(a)}")
    print(f"向量 B 维度: {len(b)}")
    if len(a) != len(b):
        raise ValueError(f"向量维度不匹配: {len(a)} != {len(b)}")
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def generate_question_vector(question):
    """生成问题的向量（可以用OpenAI的API或其他方法）"""
    # 此处可以使用OpenAI API来生成问题的embedding向量，假设我们已经获取了向量
    # 下面是一个假设的向量生成函数
    return np.random.rand(768)  # 假设返回一个768维的随机向量


def get_most_similar_question(question, vector_dir="vector_store"):
    """根据问题文本找到最相似的题目"""
    question_vector = generate_question_vector(question)

    vectors = load_vectors(vector_dir)
    max_sim = -1
    best_match = None
    for file, vector in vectors.items():
        similarity = cos_sim(question_vector, vector)
        print(f"文档 {file} 的相似度: {similarity}")  # 增加调试输出
        if similarity > max_sim:
            max_sim = similarity
            best_match = file

    if best_match is None:
        print("没有找到相关文档!")
    return best_match, max_sim

