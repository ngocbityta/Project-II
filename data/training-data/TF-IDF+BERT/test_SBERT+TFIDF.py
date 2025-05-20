import pandas as pd
import json
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from pyvi import ViTokenizer

# Load dữ liệu
df = pd.read_excel('../../raw-data/news.xlsx')

# Tiền xử lý văn bản
def preprocess_text(text):
    text = str(text).lower().strip()
    text = ViTokenizer.tokenize(text)  # Tách từ tiếng Việt
    return text

df['clean_title'] = df['title'].apply(preprocess_text)

# --- Phần 1: Tính embedding bằng SBERT ---
model = SentenceTransformer("keepitreal/vietnamese-sbert")  # Hoặc "paraphrase-multilingual-MiniLM-L12-v2"
df['sbert_embedding'] = df['clean_title'].apply(lambda x: model.encode(x))

# --- Phần 2: Tính TF-IDF Vector ---
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df['clean_title'])

# --- Xử lý truy vấn ---
query = "Du lịch sinh thái thư giãn ở biển"
query_processed = preprocess_text(query)

# Tính embedding cho truy vấn (SBERT)
query_sbert_embedding = model.encode(query_processed)

# Tính TF-IDF vector cho truy vấn
query_tfidf = vectorizer.transform([query_processed])

# --- Tính điểm riêng lẻ ---
# SBERT Score
df['sbert_score'] = df['sbert_embedding'].apply(
    lambda x: cosine_similarity([query_sbert_embedding], [x])[0][0]
)

# TF-IDF Score
tfidf_scores = cosine_similarity(query_tfidf, tfidf_matrix).flatten()
df['tfidf_score'] = tfidf_scores

# --- Kết hợp điểm theo tỉ lệ 1:1 ---
df['combined_score'] = 0.5 * df['sbert_score'] + 0.5 * df['tfidf_score']

# --- Lấy top 10 kết quả ---
top_results = df.sort_values('combined_score', ascending=False)[
    ['title', 'image', 'link', 'sbert_score', 'tfidf_score', 'combined_score']
].head(10)

# Xuất JSON
top_results.to_json('hybrid_sbert_tfidf_results.json', orient='records', force_ascii=False, indent=4)
print("Kết quả đã được lưu vào hybrid_sbert_tfidf_results.json")