import json
import os
import numpy as np
import sys
from underthesea import word_tokenize

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Hàm tính cosine similarity
def cosine_similarity(vecA, vecB):
    dot_product = np.dot(vecA, vecB)
    normA = np.linalg.norm(vecA)
    normB = np.linalg.norm(vecB)
    return dot_product / (normA * normB)

# Hàm tính vector trung bình từ dict vectors và danh sách từ
def average_sentence_vector(sentence, vector_dict):
    words = word_tokenize(sentence)
    word_vectors = [np.array(vector_dict[word]) for word in words if word in vector_dict]
    if not word_vectors:
        raise ValueError(f"Không có từ nào trong câu '{sentence}' tồn tại trong vector dictionary.")

    return np.mean(word_vectors, axis=0)

if __name__ == "__main__":
    
    if len(sys.argv) < 2:
        print(json.dumps({"error": "No sentence provided."}))
        sys.exit(1)

    sentence = sys.argv[1]
    
    try:
        vector_file_path = os.path.join(CURRENT_DIR, '../../trained-data/word2vec/vector.json')
        with open(vector_file_path, 'r') as f:
            data = json.load(f)
        
        vectors = data['vectors']

        vec1 = average_sentence_vector(sentence, vectors)

        news_file_path = os.path.join(CURRENT_DIR, '../../raw-data/news.txt')
        try:
            with open(news_file_path, 'r', encoding='utf-8') as file:
                sentences = file.readlines()
        except FileNotFoundError:
            print(json.dumps({"error": "Không tìm thấy tệp tin news.txt"}))
            sys.exit(1)

        similarities = []

        for news_sentence in sentences:
            news_sentence = news_sentence.strip()
            try:
                vec2 = average_sentence_vector(news_sentence, vectors)
                similarity = cosine_similarity(vec1, vec2)
                similarities.append({"cosine_similarity": similarity, "sentence": news_sentence})
            except ValueError:
                continue

        # Sắp xếp các câu theo cosine similarity giảm dần
        similarities.sort(reverse=True, key=lambda x: x["cosine_similarity"])
    
        print(json.dumps({"similarities": similarities}))

    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)
