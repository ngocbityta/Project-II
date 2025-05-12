import pandas as pd
import json
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from pyvi import ViTokenizer

# Load dữ liệu
df = pd.read_excel('../../raw-data/news.xlsx')

# Chọn mô hình SBERT phù hợp
model = SentenceTransformer("keepitreal/vietnamese-sbert")  # Hoặc paraphrase-multilingual-MiniLM-L12-v2

# Tiền xử lý văn bản
def preprocess_text(text):
    text = str(text).lower().strip()
    text = ViTokenizer.tokenize(text)  # Tách từ tiếng Việt
    return text

df['clean_title'] = df['title'].apply(preprocess_text)

# Tạo embedding cho tất cả tiêu đề
df['embedding'] = df['clean_title'].apply(lambda x: model.encode(x))

# Truy vấn người dùng
query = "Du lịch sinh thái thư giãn ở biển"
query_processed = preprocess_text(query)
query_embedding = model.encode(query_processed)

# Tính cosine similarity
df['score'] = df['embedding'].apply(lambda x: cosine_similarity([query_embedding], [x])[0][0])

# Lấy top 10 kết quả
top_results = df.sort_values('score', ascending=False)[['title', 'image', 'link', 'score']].head(10)

# Xuất JSON
top_results.to_json('sbert_results.json', orient='records', force_ascii=False, indent=4)
print("Kết quả đã được lưu vào sbert_results.json")