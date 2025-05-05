import json
import os
import sys
import numpy as np
from sentence_transformers import SentenceTransformer

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Hàm tính cosine similarity
def cosine_similarity(vecA, vecB):
    dot_product = np.dot(vecA, vecB)
    normA = np.linalg.norm(vecA)
    normB = np.linalg.norm(vecB)
    return dot_product / (normA * normB)

if __name__ == "__main__":
    
    if len(sys.argv) < 2:
        print(json.dumps({"error": "No sentence provided."}))
        sys.exit(1)

    sentence = sys.argv[1]
    
    try:
        # Load mô hình Sentence-BERT
        model = SentenceTransformer("keepitreal/vietnamese-sbert")

        # Encode câu truy vấn
        vec1 = model.encode(sentence)

        # Đọc nội dung từ file news.txt
        news_file_path = os.path.join(CURRENT_DIR, '../../raw-data/news.txt')
        try:
            with open(news_file_path, 'r', encoding='utf-8') as file:
                sentences = file.readlines()
        except FileNotFoundError:
            print(json.dumps({"error": "Không tìm thấy tệp tin news.txt"}))
            sys.exit(1)

        similarities = []

        # So sánh câu truy vấn với từng tiêu đề trong news.txt
        for news_sentence in sentences:
            news_sentence = news_sentence.strip()
            vec2 = model.encode(news_sentence)
            similarity = cosine_similarity(vec1, vec2)
            similarities.append({"cosine_similarity": float(similarity), "sentence": news_sentence})

        # Sắp xếp các câu theo cosine similarity giảm dần
        similarities.sort(reverse=True, key=lambda x: x["cosine_similarity"])
        top_similarities = similarities[:10]

        print(json.dumps({"similarities": top_similarities}))

    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)
