import json
import os
import numpy as np
import sys
import torch
import re
from transformers import BertTokenizer, BertForMaskedLM

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
    dot_product = np.dot(vecA, vecB)
    return dot_product / (normA * normB)

def average_sentence_vector(sentence, model, tokenizer):
    # Không dùng underthesea, chỉ dùng tokenizer của BERT
    sentence = sentence.lower()
    inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        last_hidden_states = outputs[0]
        sentence_embedding = last_hidden_states.mean(dim=1).squeeze().cpu().numpy()
    return sentence_embedding

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
        print(json.dumps({"error": "No sentence provided."}))
        sys.exit(1)

    sentence = sys.argv[1]

    try:
        model_path = os.path.join(CURRENT_DIR, '../../trained-data/bert-model')
        model = BertForMaskedLM.from_pretrained(model_path)
        tokenizer = BertTokenizer.from_pretrained(model_path)
        model.eval()

        vec1 = average_sentence_vector(sentence, model, tokenizer)

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
            if not news_sentence:
                continue
            try:
                vec2 = average_sentence_vector(news_sentence, model, tokenizer)
                similarity = cosine_similarity(vec1, vec2)
                similarities.append({"cosine_similarity": similarity, "sentence": news_sentence})
            except Exception:
                continue

        similarities.sort(key=lambda x: x["cosine_similarity"], reverse=True)
        top_similar = similarities[:20]
        
        # Tính accuracy
        accuracy = compute_accuracy(sentence, [item['sentence'] for item in top_similar])

        print(json.dumps({"similarities": top_similar, "accuracy": accuracy}, ensure_ascii=False))

    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)
