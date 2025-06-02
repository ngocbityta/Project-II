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
    return 0.0


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

        try:
            model = SentenceTransformer(model_path)
            model.eval()
        except Exception as e:
            print(json.dumps({"error": f"Failed to load SBERT model: {str(e)}"}, ensure_ascii=False))
            sys.exit(1)

        # Đọc tiêu đề từ file
        news_file_path = os.path.join(CURRENT_DIR, '../../raw-data/news.txt')
        try:
            with open(news_file_path, 'r', encoding='utf-8') as file:
                raw_sentences = file.readlines()
        except FileNotFoundError:
            print(json.dumps({"error": "Không tìm thấy tệp tin news.txt"}, ensure_ascii=False))
            sys.exit(1)

        # Làm sạch dữ liệu: bỏ trống, trùng lặp
        cleaned_sentences = list(set([s.strip() for s in raw_sentences if s.strip()]))

        if not cleaned_sentences:
            print(json.dumps({"error": "Danh sách tiêu đề rỗng sau khi lọc."}, ensure_ascii=False))
            sys.exit(1)

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Batch encode các tiêu đề
        sentence_embeddings = model.encode(
            cleaned_sentences,
            convert_to_numpy=True,
            batch_size=32,
            show_progress_bar=True,
            device=device
        )

        # Encode câu truy vấn
        query_embedding = model.encode(
            sentence,
            convert_to_numpy=True,
            device=device
        )

        # Tính cosine similarity
        similarities = [
            {
                "cosine_similarity": cosine_similarity(query_embedding, emb),
                "sentence": sent
            }
            for sent, emb in zip(cleaned_sentences, sentence_embeddings)
        ]

        # Sắp xếp theo độ tương đồng giảm dần
        similarities.sort(key=lambda x: x["cosine_similarity"], reverse=True)
        top_similar = similarities[:10]

        # Tính độ chính xác (nếu dữ liệu test phù hợp)
        accuracy = compute_accuracy(sentence, [item['sentence'] for item in top_similar])

        print(json.dumps({"similarities": top_similar, "accuracy": accuracy}, ensure_ascii=False))

    except Exception as e:
        print(json.dumps({"error": str(e)}, ensure_ascii=False))
        sys.exit(1)
