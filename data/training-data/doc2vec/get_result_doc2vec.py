import json
import os
import numpy as np
from gensim.models import Doc2Vec
from underthesea import word_tokenize
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

def cosine_similarity(vecA, vecB):
    dot_product = np.dot(vecA, vecB)
    normA = np.linalg.norm(vecA)
    normB = np.linalg.norm(vecB)
    if normA == 0 or normB == 0:
        return 0.0
    return dot_product / (normA * normB)

def get_vector(sentence, model):
    tokens = word_tokenize(sentence.lower().strip())
    inferred_vector = model.infer_vector(tokens)
    return inferred_vector

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(json.dumps({"error": "No sentence provided."}))
        sys.exit(1)
        
    sentence = sys.argv[1]

    try:
        # === Load mô hình Doc2Vec đã huấn luyện ===
        model_path = os.path.join(CURRENT_DIR, "../../trained-data/doc2vec/doc2vec.model")
        model = Doc2Vec.load(model_path)
        
        news_file_path = os.path.join(CURRENT_DIR, '../../raw-data/test_news.txt')
        try:
            with open(news_file_path, 'r', encoding='utf-8') as file:
                sentences = file.readlines()
        except FileNotFoundError:
            print(json.dumps({"error": "Không tìm thấy tệp tin test_news.txt"}))
            sys.exit(1)

        vec1 = get_vector(sentence, model)

        # === Tính cosine similarity với tất cả vector tài liệu ===
        similarities = []
        for new_sentence in sentences:
            vec2 = get_vector(new_sentence, model)
            similarity = cosine_similarity(vec1, vec2)
            similarities.append({
                "sentence": new_sentence,
                "cosine_similarity": float(similarity)
            })

        # === Sắp xếp và lấy top 10 ===
        similarities.sort(key=lambda x: x["cosine_similarity"], reverse=True)
        top_similar = similarities[:10]

        print(json.dumps({"similarities": top_similar}, ensure_ascii=False, indent=2))

    except Exception as e:
        print(json.dumps({"error": str(e)}))
        exit(1)