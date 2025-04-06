import json
import numpy as np
from underthesea import word_tokenize

# Hàm tính cosine similarity
def cosine_similarity(vecA, vecB):
    dot_product = np.dot(vecA, vecB)
    normA = np.linalg.norm(vecA)
    normB = np.linalg.norm(vecB)
    return dot_product / (normA * normB)

# Hàm tính vector trung bình từ dict vectors và danh sách từ
def average_sentence_vector(sentence, vector_dict):
    sentence = sentence.lower()
    words = word_tokenize(sentence)
    
    word_vectors = [np.array(vector_dict[word]) for word in words if word in vector_dict]

    if not word_vectors:
        raise ValueError("Không có từ nào trong câu tồn tại trong vector dictionary.")

    return np.mean(word_vectors, axis=0)

# Tải tệp JSON chứa các vectors
with open('../../trained-data/vector.json', 'r') as f:
    data = json.load(f)

vectors = data['vectors']

# Câu cần so sánh
sentence1 = 'Trump đánh Việt Nam té xác'
sentence2 = 'Việt Nam né vội'

# Tính vector trung bình cho từng câu
try:
    vec1 = average_sentence_vector(sentence1, vectors)
    vec2 = average_sentence_vector(sentence2, vectors)

    # Tính cosine similarity giữa hai câu
    similarity = cosine_similarity(vec1, vec2)
    print(f"Cosine similarity between:\n'{sentence1}'\nvs\n'{sentence2}': {similarity}")
except ValueError as e:
    print(e)
