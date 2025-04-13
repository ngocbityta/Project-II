import pandas as pd
import string
import json
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk

nltk.download('punkt')
from nltk.tokenize import word_tokenize

# Đọc file stopword chứa cả từ và cụm từ
with open('../../raw-data/stopwords.txt', 'r', encoding='utf-8') as f:
    stopwords_list = [line.strip().lower() for line in f if line.strip()]
    # Sắp xếp theo độ dài giảm dần để xử lý cụm từ trước
    stopwords_list.sort(key=lambda x: len(x.split()), reverse=True)

# Load dữ liệu từ file Excel
df = pd.read_excel('../../raw-data/news.xlsx')

# Hàm tiền xử lý văn bản
def preprocess_text(text):
    text = str(text).lower()
    text = text.translate(str.maketrans('', '', string.punctuation))  # Bỏ dấu câu
    text = re.sub(r'\s+', ' ', text)  # Bỏ khoảng trắng thừa

    for stopword in stopwords_list:
        # Loại bỏ stopword (bao gồm cụm từ)
        pattern = r'\b' + re.escape(stopword) + r'\b'
        text = re.sub(pattern, ' ', text)

    text = re.sub(r'\s+', ' ', text).strip()  # Làm sạch lại
    return text

# Áp dụng tiền xử lý cho tiêu đề
df['processed_title'] = df['title'].apply(preprocess_text)

# Tiêu đề truy vấn tìm kiếm
target = "cơ"
processed_target = preprocess_text(target)

# Vector hóa bằng TF-IDF
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df['processed_title'])
tfidf_target = vectorizer.transform([processed_target])

# Tính độ tương đồng cosine
df['similarity'] = cosine_similarity(tfidf_target, tfidf_matrix).flatten() * 100

# Sắp xếp và chọn top 10 kết quả
df = df.sort_values(by='similarity', ascending=False)
top_results = df[['title', 'image', 'link', 'similarity']].head(10)

# Ghi kết quả vào file JSON
top_results_json = top_results.to_json(orient='records', force_ascii=False)
with open('top_results_tfidf+cosine.json', 'w', encoding='utf-8') as f:
    json.dump(json.loads(top_results_json), f, ensure_ascii=False, indent=4)

print("Results have been saved to top_results_tfidf+cosine.json")
