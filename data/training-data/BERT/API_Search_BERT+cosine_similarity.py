import pandas as pd
import torch
from transformers import BertTokenizer, BertModel
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import json

# Load data từ file Excel
df = pd.read_excel('../../raw-data/news.xlsx')

# Load tokenizer và model BERT
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
model = AutoModel.from_pretrained("vinai/phobert-base")

# Hàm chuyển văn bản thành embedding từ BERT
def get_bert_embedding(text):
    tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**tokens)
    return outputs.last_hidden_state[:, 0, :].squeeze().numpy()  # Lấy CLS token

# Tạo vector embeddings cho tiêu đề
df['bert_embedding'] = df['title'].apply(get_bert_embedding)

# Tiêu đề truy vấn
target = "tâm"
target_embedding = get_bert_embedding(target)

# Tính cosine similarity giữa target và các tiêu đề
df['similarity'] = df['bert_embedding'].apply(lambda x: cosine_similarity([target_embedding], [x])[0][0] * 100)

# Sắp xếp theo độ tương tự giảm dần
df = df.sort_values(by='similarity', ascending=False)

# Lấy top kết quả
top_results = df[['title', 'image', 'link', 'similarity']].head(10)

# Chuyển thành JSON và lưu
top_results_json = top_results.to_json(orient='records', force_ascii=False)
with open('top_results_BERT+cosine.json', 'w', encoding='utf-8') as f:
    json.dump(json.loads(top_results_json), f, ensure_ascii=False, indent=4)

print("Results have been saved to top_results_BERT+cosine.json")