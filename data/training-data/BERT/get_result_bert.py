# search_from_bert_saved.py

import pandas as pd
import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
import json

# Load tokenizer và model PhoBERT
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
model = AutoModel.from_pretrained("vinai/phobert-base")
model.eval()

# Load dữ liệu đã lưu
df = pd.read_pickle('../../trained-data/bert/bert_news.pkl')
embeddings = np.load('../../trained-data/bert/bert_embeddings.npy')

# Hàm chuyển truy vấn thành embedding
def get_bert_embedding(text):
    tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**tokens)
    return outputs.last_hidden_state[:, 0, :].squeeze().numpy()

# Nhập truy vấn
query = "thành công"
query_embedding = get_bert_embedding(query)

# Tính cosine similarity
similarities = cosine_similarity([query_embedding], embeddings).flatten() * 100

# Gán similarity và sắp xếp
df['similarity'] = similarities
df = df.sort_values(by='similarity', ascending=False)

# Lấy top kết quả
top_results = df[df['similarity'] > 0][['title', 'image', 'link', 'similarity']].head(100)
top_results_json = top_results.to_json(orient='records', force_ascii=False)

# Ghi kết quả
with open('../../trained-data/bert/BERT+cosine.json', 'w', encoding='utf-8') as f:
    json.dump(json.loads(top_results_json), f, ensure_ascii=False, indent=4)

print("Top kết quả đã được lưu vào BERT+cosine.json")
