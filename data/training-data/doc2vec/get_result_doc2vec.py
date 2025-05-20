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

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(json.dumps({"error": "No sentence provided."}))
        sys.exit(1)
        
    sentence = sys.argv[1]

    try:
        # === Load mô hình Doc2Vec đã huấn luyện ===
        model_path = os.path.join(CURRENT_DIR, "../../trained-data/doc2vec/doc2vec.model")
        model = Doc2Vec.load(model_path)

        # === Load raw sentences theo tag (được lưu khi train) ===
        raw_sentences_path = os.path.join(CURRENT_DIR, "../../trained-data/doc2vec/raw_sentences.json")
        with open(raw_sentences_path, 'r', encoding='utf-8') as f:
            raw_sentences = json.load(f)

        # === Tiền xử lý câu đầu vào ===
        tokens = word_tokenize(sentence.lower())
        inferred_vector = model.infer_vector(tokens)

        # === Tính cosine similarity với tất cả vector tài liệu ===
        similarities = []
        for tag in model.dv.index_to_key:
            doc_vector = model.dv[tag]
            similarity = cosine_similarity(inferred_vector, doc_vector)
            original_text = raw_sentences.get(tag, "")
            similarities.append({
                "sentence": original_text,
                "cosine_similarity": float(similarity)
            })

        # === Sắp xếp và lấy top 10 ===
        similarities.sort(key=lambda x: x["cosine_similarity"], reverse=True)
        top_similar = similarities[:10]

        print(json.dumps({"similarities": top_similar}, ensure_ascii=False, indent=2))

    except Exception as e:
        print(json.dumps({"error": str(e)}))
        exit(1)