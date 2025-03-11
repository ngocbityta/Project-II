from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Dữ liệu mẫu
documents = [
    "Machine learning is amazing",
    "Deep learning is a subset of machine learning",
    "Natural language processing is a field of AI"
]

# Khởi tạo TF-IDF Vectorizer
vectorizer = TfidfVectorizer()

# Chuyển đổi văn bản thành ma trận TF-IDF
tfidf_matrix = vectorizer.fit_transform(documents)

# Hiển thị kết quả dưới dạng array
print(tfidf_matrix.toarray())

# Hiển thị danh sách từ vựng
print(vectorizer.get_feature_names_out())

# Vectorize the titles and the target using TF-IDF
target = "machine learning"
tfidf_target = vectorizer.transform([target])

# Calculate cosine similarity
cosine_similarities = cosine_similarity(tfidf_target, tfidf_matrix).flatten()

# Hiển thị độ tương đồng
print(cosine_similarities)

