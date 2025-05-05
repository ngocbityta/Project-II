from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# 1. Load a pretrained BERT model for sentence embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')  # Compact and fast BERT variant

# 2. Your dataset: Article titles
article_titles = [
    "Deep Learning for Natural Language Processing",
    "An Overview of Machine Learning Techniques",
    "Recent Advances in Computer Vision",
    "Natural Language Understanding with Transformers",
    "A Study on Recommender Systems",
    "Convolutional Neural Networks for Image Classification",
    "BERT for Text Classification Tasks",
    "Graph Neural Networks in Practice",
    "Improving Chatbots with Deep Learning",
    "Trends in Artificial Intelligence Research"
]

# 3. Encode all titles into BERT embeddings
title_embeddings = model.encode(article_titles)

# 4. Function to search similar article titles
def search_similar_titles(query_title, top_k=3):
    # Encode the query title
    query_embedding = model.encode([query_title])
    
    # Compute cosine similarities
    similarities = cosine_similarity(query_embedding, title_embeddings)[0]
    
    # Get top_k results
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    # Print the results
    print(f"\n Query Title: {query_title}")
    print("Top Related Article Titles:")
    for idx in top_indices:
        print(f" - {article_titles[idx]} (Score: {similarities[idx]:.4f})")

# 5. Example usage
search_similar_titles("Understanding BERT in NPL")
search_similar_titles("Neural Networks for Image Tasks")
