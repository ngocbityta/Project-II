import json
import os
import numpy as np
import sys
import glob
import re
from underthesea import word_tokenize

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

def normalize_sentence(s):
    s = s.strip().lower()
    s = re.sub(r'[.,!?]+$', '', s)
    return s

# Hàm tính cosine similarity
def cosine_similarity(vecA, vecB):
    normA = np.linalg.norm(vecA)
    normB = np.linalg.norm(vecB)
    if normA == 0 or normB == 0:
        return 0.0
    return np.dot(vecA, vecB) / (normA * normB)

# Hàm đếm số lượng cặp token có độ lệch cosine similarity > 1 - epsilon
def get_similar_tokens(sentenceA, sentenceB, vector_dict, epsilon=0.1):
    sentenceA = normalize_sentence(sentenceA)
    sentenceB = normalize_sentence(sentenceB)
    tokensA = word_tokenize(sentenceA)
    tokensB = word_tokenize(sentenceB)

    count = 0
    for wordA in tokensA:
        if wordA not in vector_dict:
            continue
        vecA = np.array(vector_dict[wordA])
        for wordB in tokensB:
            if wordB not in vector_dict:
                continue
            vecB = np.array(vector_dict[wordB])
            similarity = cosine_similarity(vecA, vecB)
            if similarity > (1 - epsilon):
                count += 1
    return count

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
        vector_file_path = os.path.join(CURRENT_DIR, '../../trained-data/word2vec/vector.json')
        with open(vector_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        vectors = data['vectors']

        news_file_path = os.path.join(CURRENT_DIR, '../../raw-data/test_news.txt')
        try:
            with open(news_file_path, 'r', encoding='utf-8') as file:
                sentences = file.readlines()
        except FileNotFoundError:
            print(json.dumps({"error": "Không tìm thấy tệp tin test_news.txt"}))
            sys.exit(1)

        similarities = []

        for news_sentence in sentences:
            try:
                similarity_count = get_similar_tokens(sentence, news_sentence, vectors, epsilon=0.5)
                similarities.append({
                    "token_pair_match_count": similarity_count,
                    "sentence": news_sentence.strip()
                })
            except ValueError:
                continue

        similarities.sort(reverse=True, key=lambda x: x["token_pair_match_count"])
        top_similar = similarities[:20]

        accuracy = compute_accuracy(sentence, [item['sentence'] for item in top_similar])

        print(json.dumps({"similarities": top_similar, "accuracy": accuracy}, ensure_ascii=False, indent=2))

    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)
