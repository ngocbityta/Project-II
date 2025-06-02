import json
import os
import numpy as np
import sys
import re
import glob
import torch
from sentence_transformers import SentenceTransformer

sys.stdout.reconfigure(encoding='utf-8')  # Thêm dòng này

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


def normalize_sentence(s):
    s = s.strip().lower()
    s = re.sub(r'[.,!?]+$', '', s)
    return s


def cosine_similarity(vecA, vecB):
    normA = np.linalg.norm(vecA)
    normB = np.linalg.norm(vecB)
    if normA == 0 or normB == 0:
        return 0.0
    return float(np.dot(vecA, vecB) / (normA * normB))


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
    return None  # Không trùng test nào thì trả về None


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(json.dumps({"error": "No sentence provided."}, ensure_ascii=False))
        sys.exit(1)

    sentence = sys.argv[1]

    try:
        model_path = os.path.join(CURRENT_DIR, '../../trained-data/sbert')
        if not os.path.isdir(model_path):
            print(json.dumps({"error": f"Model directory {model_path} not found."}, ensure_ascii=False))
            sys.exit(1)

        # Load vector và câu đã encode sẵn
        vectors_path = os.path.join(model_path, "sbert_vectors.npy")
        sentences_path = os.path.join(model_path, "sbert_sentences.json")
        if not (os.path.isfile(vectors_path) and os.path.isfile(sentences_path)):
            print(json.dumps({"error": "Không tìm thấy file vector hoặc sentences đã encode. Hãy train lại BERT."}, ensure_ascii=False))
            sys.exit(1)
        news_vectors = np.load(vectors_path)
        with open(sentences_path, "r", encoding="utf-8") as f:
            news_sentences = json.load(f)

        # Load model chỉ để encode câu truy vấn
        try:
            model = SentenceTransformer(model_path)
            model.eval()
        except Exception as e:
            print(json.dumps({"error": f"Failed to load SBERT model: {str(e)}"}, ensure_ascii=False))
            sys.exit(1)

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        query_embedding = model.encode(
            sentence,
            convert_to_numpy=True,
            device=device
        )

        # Tính cosine similarity với toàn bộ vector đã lưu
        dot_products = np.dot(news_vectors, query_embedding)
        norms = np.linalg.norm(news_vectors, axis=1) * np.linalg.norm(query_embedding)
        cosine_similarities = dot_products / (norms + 1e-8)

        # Ghép lại kết quả
        similarities = [
            {
                "cosine_similarity": float(sim),
                "sentence": sent
            }
            for sent, sim in zip(news_sentences, cosine_similarities)
        ]

        similarities.sort(key=lambda x: x["cosine_similarity"], reverse=True)
        top_similar = similarities[:10]

        accuracy = compute_accuracy(sentence, [item['sentence'] for item in top_similar])

        print(json.dumps({"similarities": top_similar, "accuracy": accuracy}, ensure_ascii=False))

    except Exception as e:
        print(json.dumps({"error": str(e)}, ensure_ascii=False))
        sys.exit(1)
