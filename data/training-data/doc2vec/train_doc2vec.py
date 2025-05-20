import os
import re
import json
import glob
import underthesea
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument

# === Cấu hình ===
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_PATH = os.path.join(CURRENT_DIR, "../../raw-data/news.txt")
MODEL_PATH = os.path.join(CURRENT_DIR, "../../trained-data/doc2vec/doc2vec.model")
STOP_WORDS_FILE = os.path.join(CURRENT_DIR, "../../raw-data/stopwords.txt")
RAW_SENTENCES_FILE = os.path.join(CURRENT_DIR, "../../trained-data/doc2vec/raw_sentences.json")

REMOVE_STOP_WORDS = True
REMOVE_PUNCTUATION = True

VECTOR_SIZE = 100
WINDOW_SIZE = 5
MIN_COUNT = 2
EPOCHS = 20
WORKERS = 4

# === Tạo thư mục nếu chưa có ===
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

# === Load stop words nếu có yêu cầu ===
stop_words = set()
if REMOVE_STOP_WORDS:
    if os.path.exists(STOP_WORDS_FILE):
        with open(STOP_WORDS_FILE, 'r', encoding='utf-8') as f:
            stop_words = set(f.read().splitlines())
    else:
        print(f"[❌] Không tìm thấy file stop words: {STOP_WORDS_FILE}")
        exit(1)

# === Load dữ liệu văn bản ===
documents = []
raw_sentences = {}
list_of_files = [INPUT_PATH] if os.path.isfile(INPUT_PATH) else glob.glob(INPUT_PATH + '/*.txt')

if not list_of_files:
    print(f"[❌] Không tìm thấy file nào trong {INPUT_PATH}")
    exit(1)

tag_index = 0
for file_path in list_of_files:
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            sentence = line.strip().lower()
            if not sentence:
                continue
            if REMOVE_PUNCTUATION:
                sentence = re.sub(r'[^\w\s\u00C0-\u1EF9]', '', sentence, flags=re.UNICODE)
            words = underthesea.word_tokenize(sentence)
            if REMOVE_STOP_WORDS:
                words = [w for w in words if w not in stop_words]
            tag = f"doc_{tag_index}"
            documents.append(TaggedDocument(words=words, tags=[tag]))
            raw_sentences[tag] = line.strip()  # Lưu câu gốc chưa xử lý
            tag_index += 1

if not documents:
    print("[❌] Không có dữ liệu hợp lệ để train.")
    exit(1)

model = Doc2Vec(
    documents=documents,
    vector_size=100,
    window=3,
    min_count=2,
    workers=4,
    epochs=20,
    dm=0,  # DBOW
    hs=1,
    negative=0,
    dbow_words=1
)

# === Lưu mô hình và dữ liệu ===
try:
    model.save(MODEL_PATH)
    with open(RAW_SENTENCES_FILE, 'w', encoding='utf-8') as f:
        json.dump(raw_sentences, f, ensure_ascii=False, indent=2)

    result = {
        "message": "✅ Training complete. Model and raw_sentences saved.",
        "status": "success"
    }
except Exception as e:
    result = {
        "error": "❌ Failed to save model or raw_sentences.",
        "details": str(e),
        "status": "fail"
    }

print(json.dumps(result, ensure_ascii=False, indent=2))
