import os
import json
import re
import string
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import save_npz

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_PATH = os.path.join(CURRENT_DIR, "../../raw-data/news.txt")
STOP_WORDS_FILE = os.path.join(CURRENT_DIR, "../../raw-data/stopwords.txt")
VECTORIZER_OUTPUT = os.path.join(CURRENT_DIR, "../../trained-data/tf-idf/tfidf_vectorizer.pkl")
MATRIX_OUTPUT = os.path.join(CURRENT_DIR, "../../trained-data/tf-idf/tfidf_matrix.npz")
PROCESSED_DF_OUTPUT = os.path.join(CURRENT_DIR, "../../trained-data/tf-idf/processed_news.pkl")

REMOVE_PUNCTUATION = True
REMOVE_STOP_WORDS = True

# === Đọc stop words nếu có yêu cầu ===
stop_words = []
if REMOVE_STOP_WORDS:
    try:
        with open(STOP_WORDS_FILE, 'r', encoding='utf-8') as f:
            stop_words = [line.strip().lower() for line in f if line.strip()]
            stop_words.sort(key=lambda x: len(x.split()), reverse=True)
    except FileNotFoundError:
        print(json.dumps({
            "error": f"File {STOP_WORDS_FILE} not found. Please ensure the file exists.",
            "status": "fail"
        }, ensure_ascii=False, indent=4))
        exit(1)

# === Hàm tiền xử lý ===
def preprocess_text(text):
    text = str(text).lower()
    if REMOVE_PUNCTUATION:
        text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text)
    if REMOVE_STOP_WORDS:
        for stopword in stop_words:
            pattern = r'\b' + re.escape(stopword) + r'\b'
            text = re.sub(pattern, ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# === Xử lý dữ liệu ===
try:
    # Đọc từng dòng trong news.txt, loại bỏ dòng trống
    with open(INPUT_PATH, 'r', encoding='utf-8') as f:
        titles = [line.strip() for line in f if line.strip()]
    df = pd.DataFrame({'title': titles})
    df['processed_title'] = df['title'].apply(preprocess_text)

    # === Vector hóa TF-IDF ===
    vectorizer = TfidfVectorizer(min_df=2, max_df=0.95, sublinear_tf=True)
    tfidf_matrix = vectorizer.fit_transform(df['processed_title'])

    # === Lưu vectorizer, matrix và dữ liệu đã xử lý ===
    with open(VECTORIZER_OUTPUT, 'wb') as f:
        pickle.dump(vectorizer, f)
    save_npz(MATRIX_OUTPUT, tfidf_matrix)
    df.to_pickle(PROCESSED_DF_OUTPUT)

    result = {
        "message": "Training TF-IDF complete. Vectorizer and matrix saved.",
        "status": "success"
    }
except Exception as e:
    result = {
        "error": "Failed to train TF-IDF or save files.",
        "details": str(e),
        "status": "fail"
    }

# === In kết quả dưới dạng JSON ===
print(json.dumps(result, ensure_ascii=False, indent=4))
