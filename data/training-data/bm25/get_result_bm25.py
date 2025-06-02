import json
import os
import numpy as np
import sys
import glob
import re
import pickle
from underthesea import word_tokenize

import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(CURRENT_DIR, "../../trained-data/bm25/bm25.model")

def normalize_sentence(s):
    s = s.strip().lower()
    s = re.sub(r'[.,!?]+$', '', s)
    return s

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
       print(json.dumps({"error": "No sentence provided."}))
       sys.exit(1)

    sentence = sys.argv[1]

    sentence_words = word_tokenize(normalize_sentence(sentence))
    
    try:
        
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)

        news_file_path = os.path.join(CURRENT_DIR, '../../raw-data/news.txt')
        try:
            with open(news_file_path, 'r', encoding='utf-8') as file:
                sentences = file.readlines()
        except FileNotFoundError:
            print(json.dumps({"error": "Không tìm thấy tệp tin news.txt"}))
            sys.exit(1)
        
        scores = model.get_scores(sentence_words)
        
        result = []

        for i, news_sentence in enumerate(sentences):
            try:
                similarity = scores[i]
                result.append({
                    "cosine_similarity": float(similarity),
                    "sentence": news_sentence.strip()
                })
            except ValueError:
                continue

        # Sắp xếp các câu theo cosine similarity giảm dần
        result.sort(reverse=True, key=lambda x: x["cosine_similarity"])
        top_similar = result[:10]
        
        # Tính accuracy
        accuracy = compute_accuracy(sentence, [item['sentence'] for item in top_similar])
    
        print(json.dumps({"similarities": top_similar, "accuracy": accuracy}, ensure_ascii=False, indent=2))

    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)
