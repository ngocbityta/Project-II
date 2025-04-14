import os
import json
import pickle
import pandas as pd
import re
import string
from scipy.sparse import load_npz
from sklearn.metrics.pairwise import cosine_similarity

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# === Đường dẫn ===
STOPWORDS_FILE = os.path.join(CURRENT_DIR, '../../raw-data/stopwords.txt')
VECTORIZER_PATH = os.path.join(CURRENT_DIR, '../../trained-data/tfidf/tfidf_vectorizer.pkl')
TFIDF_MATRIX_PATH = os.path.join(CURRENT_DIR, '../../trained-data/tfidf/tfidf_matrix.npz')
PROCESSED_DF_PATH = os.path.join(CURRENT_DIR, '../../trained-data/tfidf/processed_news.pkl')
OUTPUT_JSON_PATH = os.path.join(CURRENT_DIR, '../../trained-data/tfidf/tfidf+cosine.json')

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
            "error": f"File {STOPWORDS_FILE} not found. Please ensure the file exists.",
            "status": "fail"
        }, ensure_ascii=False, indent=4))
        exit(1)

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

# === Main logic ===
def main():
    try:
        query = "thành công"
        processed_query = preprocess_text(query)

        # Load model và dữ liệu
        with open(VECTORIZER_PATH, 'rb') as f:
            vectorizer = pickle.load(f)
        tfidf_matrix = load_npz(TFIDF_MATRIX_PATH)
        df = pd.read_pickle(PROCESSED_DF_PATH)

        # Vector hóa truy vấn và tính similarity
        tfidf_query = vectorizer.transform([processed_query])
        similarities = cosine_similarity(tfidf_query, tfidf_matrix).flatten() * 100
        df['similarity'] = similarities
        df = df.sort_values(by='similarity', ascending=False)

        # Lọc và lưu kết quả
        top_results = df[df['similarity'] > 0][['title', 'image', 'link', 'similarity']].head(100)

        with open(OUTPUT_JSON_PATH, 'w', encoding='utf-8') as f:
            json.dump(top_results.to_dict(orient='records'), f, ensure_ascii=False, indent=4)

        print(json.dumps({
            "message": "Top kết quả đã được lưu vào tfidf+cosine.json",
            "status": "success"
        }, ensure_ascii=False, indent=4))

    except Exception as e:
        print(json.dumps({
            "error": str(e),
            "status": "fail"
        }, ensure_ascii=False, indent=4))

if __name__ == "__main__":
    main()
