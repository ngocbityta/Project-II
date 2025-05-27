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

# Hàm tính vector trung bình từ dict vectors và danh sách từ
def average_sentence_vector(sentence, vector_dict):
    sentence = normalize_sentence(sentence)
    words = word_tokenize(sentence)
    word_vectors = [np.array(vector_dict[word]) for word in words if word in vector_dict]
    if not word_vectors:
        raise ValueError(f"Không có từ nào trong câu '{sentence}' tồn tại trong vector dictionary.")

    return np.mean(word_vectors, axis=0)

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

        vec1 = average_sentence_vector(sentence, vectors)

        news_file_path = os.path.join(CURRENT_DIR, '../../raw-data/news.txt')
        try:
            with open(news_file_path, 'r', encoding='utf-8') as file:
                sentences = file.readlines()
        except FileNotFoundError:
            print(json.dumps({"error": "Không tìm thấy tệp tin news.txt"}))
            sys.exit(1)

        result = []

        for news_sentence in sentences:
            try:
                vec2 = average_sentence_vector(news_sentence, vectors)
                similarity = cosine_similarity(vec1, vec2)
                result.append({"cosine_similarity": similarity, "sentence": news_sentence})
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
