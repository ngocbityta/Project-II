import pandas as pd
import torch
import re
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from pyvi import ViTokenizer 
import json
from fuzzywuzzy import process

# Đọc dữ liệu từ file Excel
df = pd.read_excel('../../raw-data/news.xlsx')

# Load tokenizer và model BERT
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
model = AutoModel.from_pretrained("vinai/phobert-base")

# Hàm chuẩn hóa văn bản
def preprocess_text(text):
    text = text.lower().strip()  # Chuyển thành chữ thường
    text = re.sub(r'[^\w\s]', '', text)  # Loại bỏ dấu câu
    text = ViTokenizer.tokenize(text)  # Tách từ bằng pyvi
    return text

# Tìm từ gần đúng nhất trong tập tiêu đề
def find_closest_word(target, candidates):
    closest_match, _ = process.extractOne(target, candidates)
    return closest_match

# Hàm lấy embedding từ BERT
def get_bert_embedding(text):
    tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**tokens)
    return outputs.last_hidden_state[:, 0, :].squeeze().numpy()  # Lấy CLS token

# Chuẩn hóa tiêu đề
df['clean_title'] = df['title'].apply(preprocess_text)

# Tạo vector embeddings từ BERT
df['bert_embedding'] = df['clean_title'].apply(get_bert_embedding)

# Tiêu đề truy vấn (có thể bị sai chính tả)
target = "du lịch sinh thái"  # Giả sử nhập sai chính tả
target = preprocess_text(target)  # Chuẩn hóa trước

# Nếu từ không có trong danh sách, tìm từ gần đúng nhất
if target not in df['clean_title'].values:
    target = find_closest_word(target, df['clean_title'].values)

# Lấy embedding của tiêu đề đã sửa lỗi
target_embedding = get_bert_embedding(target)

# Tính cosine similarity giữa target và các tiêu đề
df['bert_score'] = df['bert_embedding'].apply(lambda x: cosine_similarity([target_embedding], [x])[0][0] * 100)

# Tính điểm TF-IDF
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df['clean_title'])  # Ma trận TF-IDF từ tiêu đề đã chuẩn hóa
query_vector = vectorizer.transform([target])  # Vector TF-IDF của truy vấn đã sửa lỗi

# Tính cosine similarity giữa truy vấn và các tiêu đề
tfidf_scores = cosine_similarity(query_vector, tfidf_matrix).flatten() * 100
df['tfidf_score'] = tfidf_scores

# Hệ số trọng số giữa TF-IDF và BERT
alpha = 0.5  # Điều chỉnh mức độ ưu tiên giữa TF-IDF và BERT

# Kết hợp hai điểm số
df['final_score'] = alpha * df['tfidf_score'] + (1 - alpha) * df['bert_score']
# Sắp xếp theo điểm số cuối cùng
df = df.sort_values(by='final_score', ascending=False)

# Lấy top 10 kết quả
top_results = df[['title', 'image', 'link', 'final_score']].head(10)

# Chuyển thành JSON và lưu
top_results_json = top_results.to_json(orient='records', force_ascii=False)
with open('top_results_TFIDF_BERT.json', 'w', encoding='utf-8') as f:
    json.dump(json.loads(top_results_json), f, ensure_ascii=False, indent=4)

print(f"Results have been saved to top_results_TFIDF_BERT.json")
