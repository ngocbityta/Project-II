import os
import re
import json
from datasets import Dataset
import torch
from underthesea import word_tokenize
from sentence_transformers import SentenceTransformer, InputExample, losses, evaluation, models
from torch.utils.data import DataLoader

# Paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_PATH = os.path.join(CURRENT_DIR, "../../raw-data/news.txt")
OUTPUT_PATH = os.path.join(CURRENT_DIR, "../../trained-data/sbert")
STOP_WORDS_FILE = os.path.join(CURRENT_DIR, "../../raw-data/stopwords.txt")

REMOVE_STOP_WORDS = True
REMOVE_PUNCTUATION = True

os.makedirs(OUTPUT_PATH, exist_ok=True)

# Load stopwords
stop_words = set()
if REMOVE_STOP_WORDS:
    try:
        with open(STOP_WORDS_FILE, 'r', encoding='utf-8') as f:
            stop_words = set(f.read().splitlines())
    except FileNotFoundError:
        print(f"File {STOP_WORDS_FILE} not found. Please ensure the file exists.")
        exit(1)

# Check input file
if not os.path.isfile(INPUT_PATH):
    print(f"Input file {INPUT_PATH} not found. Please ensure the file exists.")
    exit(1)

# Read and preprocess sentences
final_sentences = []
with open(INPUT_PATH, 'r', encoding='utf-8') as f:
    sentences = f.readlines()

for i in range(len(sentences)):
    sentence = sentences[i].strip().lower()
    if REMOVE_PUNCTUATION:
        sentence = re.sub(r'[^\w\s\u00C0-\u1EF9]', '', sentence, flags=re.UNICODE)
    if REMOVE_STOP_WORDS:
        sentence = ' '.join([w for w in sentence.split() if w not in stop_words])
    sentences[i] = sentence

for sentence in sentences:
    words = word_tokenize(sentence)
    final_sentences.append(" ".join(words))

# Use first 200 sentences for quick fine-tuning/demo
small_sentences = final_sentences[:200]

# Prepare training - For SBERT fine-tuning
# Here we create dummy positive pairs by pairing each sentence with itself (can be replaced with real labeled pairs)
train_examples = [InputExample(texts=[s, s]) for s in small_sentences]

# Load a multilingual SBERT model pretrained (replace with any Vietnamese SBERT if available)
model_name = 'sentence-transformers/distiluse-base-multilingual-cased-v1'
model = SentenceTransformer(model_name)

# DataLoader and Loss function for fine-tuning
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
train_loss = losses.MultipleNegativesRankingLoss(model=model)

# Training parameters
num_epochs = 1
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Fine-tune model
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=num_epochs,
    warmup_steps=10,
    output_path=OUTPUT_PATH,
    show_progress_bar=True
)

# Save the fine-tuned model
try:
    model.save(OUTPUT_PATH)
    result = {
        "message": "SBERT fine-tuning complete. Model saved.",
        "status": "success"
    }
except Exception as e:
    result = {
        "error": "Failed to save SBERT model",
        "details": str(e),
        "status": "fail"
    }

print(json.dumps(result, ensure_ascii=False, indent=4))
