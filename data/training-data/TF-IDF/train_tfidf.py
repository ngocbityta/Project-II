import pandas as pd
import string
import re
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import save_npz
import nltk

nltk.download('punkt')

# Tiền xử lý
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

# Đọc dữ liệu
df = pd.read_excel('../../raw-data/news.xlsx')
df['processed_title'] = df['title'].apply(preprocess_text)

# Vector hóa
vectorizer = TfidfVectorizer(min_df=2, max_df=0.95, sublinear_tf=True)
tfidf_matrix = vectorizer.fit_transform(df['processed_title'])

# Lưu vectorizer và tfidf matrix
with open('../../trained-data/tfidf/tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

save_npz('../../trained-data/tfidf/tfidf_matrix.npz', tfidf_matrix)

# Lưu dữ liệu đã xử lý nếu cần
df.to_pickle('../../trained-data/tfidf/processed_news.pkl')

print("Đã lưu vectorizer, tfidf_matrix, và dataframe đã xử lý.")