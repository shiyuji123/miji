import os
import json
import numpy as np

def cos_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def load_vectors(vector_dir):
    vectors = {}
    for file in os.listdir(vector_dir):
        if file.endswith('.json'):
            with open(os.path.join(vector_dir, file), 'r') as f:
                vectors[file] = json.load(f)
    return vectors

def get_most_similar_doc(vector_dir, input_vec):
    vectors = load_vectors(vector_dir)
    best_file = None
    best_score = -1
    for fname, vec in vectors.items():
        score = cos_sim(input_vec, vec)
        if score > best_score:
            best_score = score
            best_file = fname
    return best_file, best_score
