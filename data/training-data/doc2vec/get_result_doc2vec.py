import json
import os
import numpy as np
import sys
from gensim.models import Doc2Vec
from underthesea import word_tokenize

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

        # === Load vector document đã lưu để so sánh nhãn ===
        vector_file_path = os.path.join(CURRENT_DIR, '../../trained-data/doc2vec/vector.json')
        with open(vector_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        vectors = data['vectors']

        # === Xử lý câu đầu vào ===
        tokens = word_tokenize(sentence.lower(), format='text').split()
        input_vector = model.infer_vector(tokens)

        # === Tính cosine similarity với từng document đã lưu ===
        similarities = []
        for doc_id, doc_vector in vectors.items():
            similarity = cosine_similarity(input_vector, np.array(doc_vector))
            similarities.append({
                "sentence": doc_id,
                "cosine_similarity": similarity
            })

        # === Sắp xếp theo độ tương đồng giảm dần ===
        similarities.sort(key=lambda x: x["cosine_similarity"], reverse=True)

        top_similarities = similarities[:10]
        print(json.dumps({"similarities": top_similarities}))

    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)
