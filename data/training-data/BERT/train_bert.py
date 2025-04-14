# save_bert_vectors.py

import pandas as pd
import torch
import numpy as np
import pickle
from transformers import AutoTokenizer, AutoModel

# Load dữ liệu
df = pd.read_excel('../../raw-data/news.xlsx')

# Load PhoBERT model và tokenizer
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
model = AutoModel.from_pretrained("vinai/phobert-base")
model.eval()

# Hàm tạo embedding BERT từ tiêu đề
def get_bert_embedding(text):
    tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**tokens)
    return outputs.last_hidden_state[:, 0, :].squeeze().numpy()

# Tạo embeddings
embeddings = []
for title in df['title']:
    embeddings.append(get_bert_embedding(title))

# Convert về ndarray để lưu
embeddings_array = np.array(embeddings)

# Lưu embedding và dataframe
np.save('../../trained-data/bert/bert_embeddings.npy', embeddings_array)
df.to_pickle('../../trained-data/bert/bert_news.pkl')

print("Đã lưu BERT embeddings và DataFrame.")
