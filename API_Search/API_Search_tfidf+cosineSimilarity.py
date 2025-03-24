import pandas as pd
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import json

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')

# Load data from Excel file
df = pd.read_excel('../crawler/news.xlsx')

# Preprocess text
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Tokenize
    tokens = word_tokenize(text)
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Apply preprocessing to the titles in the dataframe
df['processed_title'] = df['title'].apply(preprocess_text)

# Preprocess the target
target = "chiến"
processed_target = preprocess_text(target)

# Vectorize the titles and the target using TF-IDF
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df['processed_title'])
tfidf_target = vectorizer.transform([processed_target])

# Calculate cosine similarity
cosine_similarities = cosine_similarity(tfidf_target, tfidf_matrix).flatten()

# Rank the titles based on similarity
df['similarity'] = cosine_similarities * 100
df = df.sort_values(by='similarity', ascending=False)

# Display the top results
top_results = df[['title', 'image', 'link', 'similarity']].head(10)

# Filter and get the top results with similarity greater than 1%
#top_results = df[df['similarity'] > 1][['title', 'image', 'link', 'similarity']].head(10)

# Convert to JSON and save to file with pretty formatting
top_results_json = top_results.to_json(orient='records', force_ascii=False)
with open('top_results_tfidf+cosine.json', 'w', encoding='utf-8') as f:
    json.dump(json.loads(top_results_json), f, ensure_ascii=False, indent=4)

print("Results have been saved to top_results_tfidf+cosine.json")