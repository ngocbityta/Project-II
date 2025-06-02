import json
import os
import numpy as np
import sys
import glob
import re
from underthesea import word_tokenize
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
    normA = np.linalg.norm(vecA)
    normB = np.linalg.norm(vecB)
    if normA == 0 or normB == 0:
        return 0.0
    return np.dot(vecA, vecB) / (normA * normB)

# Dùng vectors toàn cục
def average_sentence_vector(sentence):
    sentence = normalize_sentence(sentence)
    words = word_tokenize(sentence)
    sentence = ' '.join([word for word in sentence.split() if word not in stop_words])
    word_vectors = [np.array(vectors[word]) for word in words if word in vectors]
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
    vectors = {}  
    
    if len(sys.argv) < 2:
       print(json.dumps({"error": "No sentence provided."}))
       sys.exit(1)

    sentence = sys.argv[1]
    
    try:
        # Load vectors và chuyển sang numpy array một lần
        vector_file_path = os.path.join(CURRENT_DIR, '../../trained-data/word2vec/vector.json')
        with open(vector_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            vectors = {k: np.array(v) for k, v in data['vectors'].items()}
        vec1 = average_sentence_vector(sentence)

        news_file_path = os.path.join(CURRENT_DIR, '../../raw-data/news.txt')
        try:
            with open(news_file_path, 'r', encoding='utf-8') as file:
                sentences = file.readlines()
        except FileNotFoundError:
            print(json.dumps({"error": "Không tìm thấy tệp tin news.txt"}))
            sys.exit(1)

        result = []
        for news_sentence in sentences:
            if get_common_token_count(sentence, news_sentence) <= 1:
                continue
            try:
                vec2 = average_sentence_vector(news_sentence)
                similarity = cosine_similarity(vec1, vec2)
                result.append({"cosine_similarity": similarity, "sentence": news_sentence})
            except ValueError:
                continue

        result.sort(reverse=True, key=lambda x: x["cosine_similarity"])
        top_similar = [item for item in result if item["cosine_similarity"] > 0.5][:20]

        accuracy = compute_accuracy(sentence, [item['sentence'] for item in top_similar])
        print(json.dumps({"similarities": top_similar, "accuracy": accuracy}, ensure_ascii=False, indent=2))

    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)
