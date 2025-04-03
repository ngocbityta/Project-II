import json
import numpy as np

# Hàm tính cosine similarity
def cosine_similarity(vecA, vecB):
    dot_product = np.dot(vecA, vecB)
    normA = np.linalg.norm(vecA)
    normB = np.linalg.norm(vecB)
    return dot_product / (normA * normB)

# Tải tệp JSON chứa các vectors
with open('vectors.json', 'r') as f:
    data = json.load(f)

vectors = data['vectors']

# Lấy vector của hai từ cần tính tương đồng
word1 = 'king'
word2 = 'queen'

# Kiểm tra xem từ có trong vectors không
if word1 in vectors and word2 in vectors:
    vec1 = np.array(vectors[word1])
    vec2 = np.array(vectors[word2])

    # Tính cosine similarity
    similarity = cosine_similarity(vec1, vec2)
    print(f"Cosine similarity between '{word1}' and '{word2}': {similarity}")
else:
    print("One or both words are not in the vocabulary.")
