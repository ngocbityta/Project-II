import json
import os
import numpy as np
import glob
import re
from gensim.models import Doc2Vec
from underthesea import word_tokenize
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
STOP_WORDS_FILE = os.path.join(CURRENT_DIR, "../../raw-data/stopwords.txt")
REMOVE_STOP_WORDS = True

stop_words = set()
if REMOVE_STOP_WORDS:
    try:
        with open(STOP_WORDS_FILE, 'r', encoding='utf-8') as f:
            stop_words = set(f.read().splitlines())
    except FileNotFoundError:
        print(f"File {STOP_WORDS_FILE} not found. Please ensure the file exists.")
        exit(1)

def normalize_sentence(s):
    # Loại bỏ ký tự xuống dòng, tab
    s = s.replace('\n', ' ').replace('\t', ' ')
    
    # Loại bỏ khoảng trắng đầu/cuối và chuyển về chữ thường
    s = s.strip().lower()
    
    # Loại bỏ toàn bộ ký tự không phải chữ cái, số hoặc khoảng trắng
    s = re.sub(r'[^\w\s]', '', s, flags=re.UNICODE)
    
    # Loại bỏ stop words
    s = ' '.join([word for word in s.split() if word not in stop_words])
    
    return s

def get_common_token_count(sentence1, sentence2):
    sentence1 = normalize_sentence(sentence1)
    sentence2 = normalize_sentence(sentence2)
    tokens1 = set(word_tokenize(sentence1))
    tokens2 = set(word_tokenize(sentence2))
    if len(tokens1) <= 3 or len(tokens2) <= 3:
        return 2
    return len(tokens1 & tokens2)

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
    return None  # Không trùng test nào thì trả về None


def get_vector(sentence):
    tokens = word_tokenize(normalize_sentence(sentence))
    inferred_vector = model.infer_vector(tokens, alpha=0.025, min_alpha=0.0001, epochs=50)
    return inferred_vector

if __name__ == "__main__":
    model = {}
    if len(sys.argv) < 2:
        print(json.dumps({"error": "No sentence provided."}))
        sys.exit(1)
        
    sentence = sys.argv[1]

    try:
        # === Load mô hình Doc2Vec đã huấn luyện ===
        model_path = os.path.join(CURRENT_DIR, "../../trained-data/doc2vec/doc2vec.model")
        model = Doc2Vec.load(model_path)
        
        news_file_path = os.path.join(CURRENT_DIR, '../../raw-data/news.txt')
        try:
            with open(news_file_path, 'r', encoding='utf-8') as file:
                sentences = file.readlines()
        except FileNotFoundError:
            print(json.dumps({"error": "Không tìm thấy tệp tin news.txt"}))
            sys.exit(1)

        vec1 = get_vector(sentence)

        # === Tính cosine similarity với tất cả vector tài liệu ===
        result = []
        for new_sentence in sentences:
            if get_common_token_count(sentence, new_sentence) == 0:
                continue
            vec2 = get_vector(new_sentence)
            similarity = cosine_similarity(vec1, vec2)
            result.append({
                "sentence": new_sentence,
                "cosine_similarity": float(similarity)
            })

        result.sort(key=lambda x: x["cosine_similarity"], reverse=True)
        top_similar = [item for item in result if item["cosine_similarity"] > 0.53][:15]
        
        # === Tính accuracy ===
        accuracy = compute_accuracy(sentence, [s["sentence"] for s in top_similar])

        print(json.dumps({"similarities": top_similar, "accuracy": accuracy}, ensure_ascii=False))

    except Exception as e:
        print(json.dumps({"error": str(e)}))
        exit(1)