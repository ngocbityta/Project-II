import os
import glob
import re
import json
from transformers import BertTokenizer, BertForMaskedLM, Trainer, TrainingArguments
from datasets import Dataset
import torch
import underthesea

# Define paths and parameters
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_PATH = os.path.join(CURRENT_DIR, "../../raw-data/news.txt")
OUTPUT_PATH = os.path.join(CURRENT_DIR, "../../trained-data/bert-model")

STOP_WORDS_FILE = os.path.join(CURRENT_DIR, "../../raw-data/stopwords.txt")
REMOVE_STOP_WORDS = True
REMOVE_PUNCTUATION = True

# Make sure output directory exists
os.makedirs(OUTPUT_PATH, exist_ok=True)

# === Read stopwords if required ===
stop_words = set()
if REMOVE_STOP_WORDS:
    try:
        with open(STOP_WORDS_FILE, 'r', encoding='utf-8') as f:
            stop_words = set(f.read().splitlines())
    except FileNotFoundError:
        print(f"File {STOP_WORDS_FILE} not found. Please ensure the file exists.")
        exit(1)

# Check if input file exists
if not os.path.isfile(INPUT_PATH):
    print(f"Input file {INPUT_PATH} not found. Please ensure the file exists.")
    exit(1)

# === Read and preprocess text files ===
final_sentences = []
listOfFiles = [INPUT_PATH] if os.path.isfile(INPUT_PATH) else glob.glob(INPUT_PATH + '/*.txt')

for file in listOfFiles:
    with open(file, 'r', encoding='utf-8') as f:
        sentences = f.readlines()

    for i in range(len(sentences)):
        sentence = sentences[i].strip().lower()
        if REMOVE_PUNCTUATION:
            # Remove punctuation but keep unicode letters (Vietnamese accents)
            sentence = re.sub(r'[^\w\s\u00C0-\u1EF9]', '', sentence, flags=re.UNICODE)
        if REMOVE_STOP_WORDS:
            sentence = ' '.join([word for word in sentence.split() if word not in stop_words])
        sentences[i] = sentence

    for sentence in sentences:
        words = underthesea.word_tokenize(sentence)
        final_sentences.append(" ".join(words))

# === Tokenizer for BERT ===
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

# === Masking tokens and creating labels ===
def mask_tokens(inputs, tokenizer, probability=0.15):
    """Prepare masked tokens for MLM."""
    labels = inputs.clone()
    mask = torch.rand(inputs.shape) < probability
    labels[~mask] = -100  # Ignore non-masked tokens in the loss calculation
    inputs[mask] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)  # Replace with [MASK] token
    return inputs, labels

# === Tokenize the sentences and apply masking ===
def tokenize_function(examples):
    encoding = tokenizer(examples['text'], padding='max_length', truncation=True, max_length=512)
    input_ids = torch.tensor(encoding['input_ids'])
    input_ids, labels = mask_tokens(input_ids, tokenizer)
    encoding['input_ids'] = input_ids.tolist()
    encoding['labels'] = labels.tolist()
    return encoding

# === Create a dataset from the text data ===
dataset = Dataset.from_dict({"text": final_sentences})

# Tokenize the dataset
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# === Model and Trainer Setup ===
model = BertForMaskedLM.from_pretrained('bert-base-multilingual-cased')

training_args = TrainingArguments(
    output_dir=OUTPUT_PATH,
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=8,  # Adjust batch size if out of memory
    save_steps=10_000,
    save_total_limit=2,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
)

# === Train the BERT model ===
trainer.train()

# === Save the trained model ===
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

# Output result as JSON
print(json.dumps(result, ensure_ascii=False, indent=4))
