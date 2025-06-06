import os
import re
import json
import shutil
import sys
import numpy as np
from datasets import Dataset
import torch
from underthesea import word_tokenize
from sentence_transformers import SentenceTransformer, InputExample, losses, evaluation, models
from torch.utils.data import DataLoader

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_PATH = os.path.join(CURRENT_DIR, "../../raw-data/news.txt")
OUTPUT_PATH = os.path.join(CURRENT_DIR, "../../trained-data/sbert")
STOP_WORDS_FILE = os.path.join(CURRENT_DIR, "../../raw-data/stopwords.txt")

REMOVE_STOP_WORDS = True
REMOVE_PUNCTUATION = True

def safe_print_json(obj):
    print(json.dumps(obj, ensure_ascii=False, indent=4))

try:
    # Xóa model cũ nếu tồn tại để tránh lỗi ghi đè trên Windows
    if os.path.exists(OUTPUT_PATH):
        try:
            shutil.rmtree(OUTPUT_PATH)
        except Exception as e:
            safe_print_json({
                "error": "Failed to remove old SBERT model directory",
                "details": str(e),
                "status": "fail"
            })
            sys.exit(1)

    os.makedirs(OUTPUT_PATH, exist_ok=True)

    # Load stopwords
    stop_words = set()
    if REMOVE_STOP_WORDS:
        try:
            with open(STOP_WORDS_FILE, 'r', encoding='utf-8') as f:
                stop_words = set(f.read().splitlines())
        except FileNotFoundError:
            safe_print_json({
                "error": f"File {STOP_WORDS_FILE} not found. Please ensure the file exists.",
                "status": "fail"
            })
            sys.exit(1)

    # Check input file
    if not os.path.isfile(INPUT_PATH):
        safe_print_json({
            "error": f"Input file {INPUT_PATH} not found. Please ensure the file exists.",
            "status": "fail"
        })
        sys.exit(1)

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

    # Dùng toàn bộ dữ liệu để train (không lấy tập nhỏ nữa)
    train_examples = [InputExample(texts=[s, s]) for s in final_sentences]

    # Load a multilingual SBERT model pretrained (replace with any Vietnamese SBERT if available)
    model_name = "VoVanPhuc/sup-SimCSE-VietNamese-phobert-base"
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
        # Encode toàn bộ news.txt thành vector và lưu lại
        news_file = INPUT_PATH
        with open(news_file, 'r', encoding='utf-8') as f:
            news_sentences = [line.strip() for line in f if line.strip()]
        # Loại trùng lặp, rỗng
        news_sentences = list(dict.fromkeys(news_sentences))
        news_vectors = model.encode(news_sentences, convert_to_numpy=True, batch_size=32, show_progress_bar=True, device=device)
        # Lưu vector nhị phân và câu gốc
        np.save(os.path.join(OUTPUT_PATH, "sbert_vectors.npy"), news_vectors)
        with open(os.path.join(OUTPUT_PATH, "sbert_sentences.json"), "w", encoding="utf-8") as f:
            json.dump(news_sentences, f, ensure_ascii=False, indent=2)
        result = {
            "message": "SBERT fine-tuning complete. Model and vectors saved.",
            "status": "success"
        }
    except Exception as e:
        result = {
            "error": "Failed to save SBERT model or vectors",
            "details": str(e),
            "status": "fail"
        }

    safe_print_json(result)

except Exception as e:
    safe_print_json({
        "error": "Exception during SBERT training",
        "details": str(e),
        "status": "fail"
    })
    sys.exit(1)
