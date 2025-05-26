import os
import re
import json
from transformers import DistilBertTokenizer, DistilBertForMaskedLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import Dataset
import torch

# Paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_PATH = os.path.join(CURRENT_DIR, "../../raw-data/news.txt")
OUTPUT_PATH = os.path.join(CURRENT_DIR, "../../trained-data/bert-model")

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

# Read and preprocess
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
    words = underthesea.word_tokenize(sentence)
    final_sentences.append(" ".join(words))

<<<<<<< Updated upstream
# Use only first 200 sentences for ultra-fast training
small_sentences = final_sentences[:200]
=======
    # Không dùng underthesea, chỉ giữ nguyên câu đã xử lý
    final_sentences.extend([s for s in sentences if s])
>>>>>>> Stashed changes

dataset = Dataset.from_dict({"text": small_sentences})

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-multilingual-cased')

def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=32)

tokenized_datasets = dataset.map(tokenize_function, batched=True, num_proc=4)

model = DistilBertForMaskedLM.from_pretrained('distilbert-base-multilingual-cased')

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
model.to(device)

import torch

use_fp16 = torch.cuda.is_available()  # Only use fp16 if CUDA GPU available

training_args = TrainingArguments(
    output_dir=OUTPUT_PATH,
    overwrite_output_dir=True,
    num_train_epochs=1,
    max_steps=20,
    per_device_train_batch_size=64,
    save_steps=1000000,
    logging_steps=1000,
    fp16=use_fp16,  # Enable fp16 only if CUDA GPU
    load_best_model_at_end=False,
)


data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
    data_collator=data_collator,
)

trainer.train()

try:
    model.save_pretrained(OUTPUT_PATH)
    tokenizer.save_pretrained(OUTPUT_PATH)
    result = {
        "message": "Training complete. BERT model saved.",
        "status": "success"
    }
except Exception as e:
    result = {
        "error": "Failed to save BERT model",
        "details": str(e),
        "status": "fail"
    }

print(json.dumps(result, ensure_ascii=False, indent=4))
