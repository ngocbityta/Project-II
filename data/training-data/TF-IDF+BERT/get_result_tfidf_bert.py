import sys
import io
import json
import os
import re
import pickle
import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from sklearn.metrics.pairwise import cosine_similarity
import glob
import torch
from sentence_transformers import SentenceTransformer

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def normalize_sentence(s):
    return re.sub(r'[^\w\s]', '', s.strip().lower())

def compute_f1_score(true_sentences, predicted_sentences):
    true_set = set([normalize_sentence(s) for s in true_sentences])
    pred_set = set([normalize_sentence(s) for s in predicted_sentences])
    true_positives = len(true_set & pred_set)
    precision = true_positives / len(pred_set) if pred_set else 0
    recall = true_positives / len(true_set) if true_set else 0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)

def compute_accuracy(sentence, predicted_sentences, CURRENT_DIR):
    valid_data_dir = os.path.join(CURRENT_DIR, '../../valid-data')
    json_files = glob.glob(os.path.join(valid_data_dir, 'test_*.json'))
    for file_path in json_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if normalize_sentence(data.get('searchText')) == normalize_sentence(sentence):
                return compute_f1_score(data.get('result'), predicted_sentences)
    return None

def preprocess_text(text, stopwords_list, remove_punct=True, remove_stop=True):
    import string
    text = str(text).lower()
    if remove_punct:
        text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text)
    if remove_stop:
        for stopword in stopwords_list:
            pattern = r'\b' + re.escape(stopword) + r'\b'
            text = re.sub(pattern, ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(json.dumps({"error": "No sentence provided."}))
        sys.exit(1)
    sentence = sys.argv[1]
    alpha = float(sys.argv[2]) if len(sys.argv) > 2 else 0.7

    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

    # TF-IDF paths
    STOPWORDS_FILE = os.path.join(CURRENT_DIR, '../../raw-data/stopwords.txt')
    VECTORIZER_PATH = os.path.join(CURRENT_DIR, '../../trained-data/tf-idf/tfidf_vectorizer.pkl')
    TFIDF_MATRIX_PATH = os.path.join(CURRENT_DIR, '../../trained-data/tf-idf/tfidf_matrix.npz')
    PROCESSED_DF_PATH = os.path.join(CURRENT_DIR, '../../trained-data/tf-idf/processed_news.pkl')

    # SBERT paths
    SBERT_DIR = os.path.join(CURRENT_DIR, '../../trained-data/sbert')
    SBERT_VECTORS_PATH = os.path.join(SBERT_DIR, "sbert_vectors.npy")
    SBERT_SENTENCES_PATH = os.path.join(SBERT_DIR, "sbert_sentences.json")

    try:
        # Load stopwords
        stopwords_list = []
        try:
            with open(STOPWORDS_FILE, 'r', encoding='utf-8') as f:
                stopwords_list = [line.strip().lower() for line in f if line.strip()]
                stopwords_list.sort(key=lambda x: len(x.split()), reverse=True)
        except FileNotFoundError:
            print(json.dumps({"error": f"File {STOPWORDS_FILE} not found."}))
            sys.exit(1)

        # Load TF-IDF
        with open(VECTORIZER_PATH, 'rb') as f:
            vectorizer = pickle.load(f)
        tfidf_matrix = load_npz(TFIDF_MATRIX_PATH)
        df = pd.read_pickle(PROCESSED_DF_PATH)
        news_titles = df['title'].tolist()

        # Load SBERT
        if not (os.path.isdir(SBERT_DIR) and os.path.isfile(SBERT_VECTORS_PATH) and os.path.isfile(SBERT_SENTENCES_PATH)):
            print(json.dumps({"error": "Không tìm thấy SBERT model hoặc vectors."}))
            sys.exit(1)
        news_vectors = np.load(SBERT_VECTORS_PATH)
        with open(SBERT_SENTENCES_PATH, "r", encoding="utf-8") as f:
            sbert_sentences = json.load(f)
        model = SentenceTransformer(SBERT_DIR)
        model.eval()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Tiền xử lý truy vấn
        processed_query = preprocess_text(sentence, stopwords_list)
        tfidf_query = vectorizer.transform([processed_query])
        query_embedding = model.encode(sentence, convert_to_numpy=True, device=device)

        # Tính similarity
        tfidf_similarities = cosine_similarity(tfidf_query, tfidf_matrix).flatten()
        bert_similarities = np.dot(news_vectors, query_embedding) / (np.linalg.norm(news_vectors, axis=1) * np.linalg.norm(query_embedding) + 1e-8)

        # Kết hợp
        combined = []
        for idx, sent in enumerate(news_titles):
            sent = sent.strip()
            tfidf_score = tfidf_similarities[idx] if idx < len(tfidf_similarities) else 0.0
            bert_score = bert_similarities[idx] if idx < len(bert_similarities) else 0.0
            final_score = alpha * tfidf_score + (1 - alpha) * bert_score
            combined.append({
                "sentence": sent,
                "final_score": float(final_score)
            })

        combined.sort(key=lambda x: x["final_score"], reverse=True)
        top_items = [item for item in combined if item["final_score"] > 0][:15]
        top_sentences = [item["sentence"] for item in top_items]

        accuracy = compute_accuracy(sentence, top_sentences, CURRENT_DIR)

        print(json.dumps({
            "similarities": top_items,
            "accuracy": accuracy
        }, ensure_ascii=False))
    except Exception as e:
        print(json.dumps({
            "similarities": [],
            "accuracy": 0.0,
            "error": str(e)
        }))
        sys.exit(1)
