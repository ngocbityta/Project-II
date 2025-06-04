import os
import json
import pickle
import pandas as pd
import re
import string
import sys
import glob
from scipy.sparse import load_npz
from sklearn.metrics.pairwise import cosine_similarity

import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# === Đường dẫn ===
STOPWORDS_FILE = os.path.join(CURRENT_DIR, '../../raw-data/stopwords.txt')
VECTORIZER_PATH = os.path.join(CURRENT_DIR, '../../trained-data/tf-idf/tfidf_vectorizer.pkl')
TFIDF_MATRIX_PATH = os.path.join(CURRENT_DIR, '../../trained-data/tf-idf/tfidf_matrix.npz')
PROCESSED_DF_PATH = os.path.join(CURRENT_DIR, '../../trained-data/tf-idf/processed_news.pkl')

REMOVE_STOP_WORDS = True
REMOVE_PUNCTUATION = True

# === Load stopwords ===
stopwords_list = []
if REMOVE_STOP_WORDS:
    try:
        with open(STOPWORDS_FILE, 'r', encoding='utf-8') as f:
            stopwords_list = [line.strip().lower() for line in f if line.strip()]
            stopwords_list.sort(key=lambda x: len(x.split()), reverse=True)
    except FileNotFoundError:
        print(json.dumps({
            "error": f"File {STOPWORDS_FILE} not found. Please ensure the file exists."
        }, ensure_ascii=False))
        sys.exit(1)
        
def normalize_sentence(s):
    s = s.strip().lower()
    s = re.sub(r'[^\w\s]', '', s, flags=re.UNICODE)
    return s

# === Tiền xử lý truy vấn ===
def preprocess_text(text):
    text = str(text).lower()
    if REMOVE_PUNCTUATION:
        text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text)
    if REMOVE_STOP_WORDS:
        for stopword in stopwords_list:
            pattern = r'\b' + re.escape(stopword) + r'\b'
            text = re.sub(pattern, ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

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


# === Main logic ===
def main():
    if len(sys.argv) < 2:
        print(json.dumps({"error": "No sentence provided."}))
        sys.exit(1)

    query = sys.argv[1]

    try:
        processed_query = preprocess_text(query)

        # Load vectorizer, tfidf matrix, và dataframe
        with open(VECTORIZER_PATH, 'rb') as f:
            vectorizer = pickle.load(f)
        tfidf_matrix = load_npz(TFIDF_MATRIX_PATH)
        df = pd.read_pickle(PROCESSED_DF_PATH)

        # Tính similarity
        tfidf_query = vectorizer.transform([processed_query])
        similarities = cosine_similarity(tfidf_query, tfidf_matrix).flatten()

        result = []
        for idx, sim in enumerate(similarities):
            if sim > 0:
                result.append({
                    "cosine_similarity": float(sim),
                    "sentence": df.iloc[idx]["title"]
                })

        # Sắp xếp giảm dần
        result.sort(key=lambda x: x["cosine_similarity"], reverse=True)
        
        top_similar = [item for item in result if item["cosine_similarity"] > 0.25][:15]
        
        # Tính accuracy
        accuracy = compute_accuracy(query, [item['sentence'] for item in top_similar])

        print(json.dumps({"similarities": top_similar, "accuracy": accuracy}, ensure_ascii=False))

    except Exception as e:
        print(json.dumps({"error": str(e)}, ensure_ascii=False))
        sys.exit(1)

if __name__ == "__main__":
    main()
