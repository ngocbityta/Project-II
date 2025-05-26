import json
import os
import numpy as np
import sys
import torch
from transformers import BertTokenizer, BertForMaskedLM

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

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
        top_similar = similarities[:10]
        print(json.dumps({"similarities": top_similar}, ensure_ascii=False))

    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)
