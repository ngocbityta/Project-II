# search_from_saved.py

import pickle
import pandas as pd
import re
import string
from scipy.sparse import load_npz
from sklearn.metrics.pairwise import cosine_similarity
import nltk
nltk.download('punkt')

# Load stopwords
with open('../../raw-data/stopwords.txt', 'r', encoding='utf-8') as f:
    stopwords_list = [line.strip().lower() for line in f if line.strip()]
    stopwords_list.sort(key=lambda x: len(x.split()), reverse=True)

def preprocess_text(text):
    text = str(text).lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text)
    for stopword in stopwords_list:
        pattern = r'\b' + re.escape(stopword) + r'\b'
        text = re.sub(pattern, ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Load dữ liệu đã lưu
with open('../../trained-data/tfidf/tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

tfidf_matrix = load_npz('../../trained-data/tfidf/tfidf_matrix.npz')
df = pd.read_pickle('../../trained-data/tfidf/processed_news.pkl')

# Nhập truy vấn
query = "thành công"
processed_query = preprocess_text(query)
tfidf_query = vectorizer.transform([processed_query])

# Tính độ tương đồng
similarities = cosine_similarity(tfidf_query, tfidf_matrix).flatten() * 100
df['similarity'] = similarities
df = df.sort_values(by='similarity', ascending=False)

# Lọc và lưu kết quả
top_results = df[df['similarity'] > 0][['title', 'image', 'link', 'similarity']].head(100)
top_results.to_json('../../trained-data/tfidf/tfidf+cosine.json', orient='records', force_ascii=False, indent=4)

print("Top kết quả đã được lưu vào tfidf+cosine.json")
