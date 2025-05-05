import pandas as pd
import re
import json
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from pyvi import ViTokenizer
from sentence_transformers import SentenceTransformer
from fuzzywuzzy import process

# Đọc dữ liệu từ file Excel
df = pd.read_excel('../../raw-data/news.xlsx')

# Load sentence-transformer model (đa ngôn ngữ, hỗ trợ tiếng Việt)
model = SentenceTransformer('all-MiniLM-L6-v2')

# Hàm chuẩn hóa văn bản
def preprocess_text(text):
    text = str(text).lower().strip()
    text = re.sub(r'[^\w\s]', '', text)  # Loại bỏ dấu câu
    text = ViTokenizer.tokenize(text)    # Tách từ bằng PyVi
    return text

# Tìm từ gần đúng nhất trong tập tiêu đề
def find_closest_word(target, candidates):
    closest_match, _ = process.extractOne(target, candidates)
    return closest_match

# Hàm tạo embedding từ sentence-transformer
def get_embedding(text):
    return model.encode(text, convert_to_numpy=True)

# Chuẩn hóa tiêu đề
df['clean_title'] = df['title'].apply(preprocess_text)

# Tạo embedding từ tiêu đề đã chuẩn hóa
df['embedding'] = df['clean_title'].apply(get_embedding)

# Tiêu đề truy vấn (có thể sai chính tả)
target = "Căng thẳng nga mỹ"
target = preprocess_text(target)

# Nếu target không giống hoàn toàn, tìm gần đúng nhất
if target not in df['clean_title'].values:
    target = find_closest_word(target, df['clean_title'].values)

# Tạo embedding cho truy vấn
target_embedding = get_embedding(target)

# Tính cosine similarity với embedding
df['bert_score'] = df['embedding'].apply(lambda x: cosine_similarity([target_embedding], [x])[0][0] * 100)

# TF-IDF vector hóa
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df['clean_title'])
query_vector = vectorizer.transform([target])

# Tính điểm TF-IDF cosine similarity
tfidf_scores = cosine_similarity(query_vector, tfidf_matrix).flatten() * 100
df['tfidf_score'] = tfidf_scores

# Trọng số kết hợp TF-IDF và BERT
alpha = 0.5
df['final_score'] = alpha * df['tfidf_score'] + (1 - alpha) * df['bert_score']

# Sắp xếp và lấy top 10 kết quả
top_results = df.sort_values(by='final_score', ascending=False)[['title', 'image', 'link', 'final_score']].head(10)

# Xuất ra JSON
top_results_json = top_results.to_json(orient='records', force_ascii=False)
with open('test_TFIDF+BERT.json', 'w', encoding='utf-8') as f:
    json.dump(json.loads(top_results_json), f, ensure_ascii=False, indent=4)

print("Results have been saved to test_TFIDF+BERT.json")
