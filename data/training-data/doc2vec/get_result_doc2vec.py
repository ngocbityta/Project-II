import json
import os
import numpy as np
import glob
import re
from gensim.models import Doc2Vec
from underthesea import word_tokenize
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

def normalize_sentence(s):
    s = s.strip().lower()
    s = re.sub(r'[.,!?]+$', '', s)
    return s

def cosine_similarity(vecA, vecB):
    dot_product = np.dot(vecA, vecB)
    normA = np.linalg.norm(vecA)
    normB = np.linalg.norm(vecB)
    if normA == 0 or normB == 0:
        return 0.0
    return dot_product / (normA * normB)

def compute_f1_score(true_sentences, predicted_sentences):
    true_set = set([normalize_sentence(s) for s in true_sentences])
    pred_set = set([normalize_sentence(s) for s in predicted_sentences])

    true_positives = len(true_set & pred_set)
    precision = true_positives / len(pred_set) if pred_set else 0
    recall = true_positives / len(true_set) if true_set else 0

    if precision + recall == 0:
        return 0.0

    f1 = 2 * precision * recall / (precision + recall)
    return f1

def compute_accuracy(sentence, predicted_sentences):
    valid_data_dir = os.path.join(CURRENT_DIR, '../../valid-data')
    json_files = glob.glob(os.path.join(valid_data_dir, 'test_*.json'))
    for file_path in json_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if normalize_sentence(data.get('searchText')) == normalize_sentence(sentence):
                return compute_f1_score(data.get('result'), predicted_sentences)
    return 0.0


def get_vector(sentence, model):
    tokens = word_tokenize(normalize_sentence(sentence))
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
        top_similar = similarities[:15]
        
        # === Tính accuracy ===
        accuracy = compute_accuracy(sentence, [s["sentence"] for s in top_similar])

        print(json.dumps({"similarities": top_similar, "accuracy": accuracy}, ensure_ascii=False, indent=2))

    except Exception as e:
        print(json.dumps({"error": str(e)}))
        exit(1)