import json
import glob
import os
import re
import underthesea
from rank_bm25 import BM25Okapi
import pickle

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_PATH = os.path.join(CURRENT_DIR, "../../raw-data/news.txt")
MODEL_PATH = os.path.join(CURRENT_DIR, "../../trained-data/bm25/bm25.model")
STOP_WORDS_FILE = os.path.join(CURRENT_DIR, "../../raw-data/stopwords.txt")

REMOVE_STOP_WORDS = True
REMOVE_PUNCTUATION = True

# === Xử lý danh sách file ===
listOfFiles = [INPUT_PATH] if os.path.isfile(INPUT_PATH) else glob.glob(INPUT_PATH + '/*.txt')

# === Đọc stop words nếu có yêu cầu ===
stop_words = set()
if REMOVE_STOP_WORDS:
    try:
        with open(STOP_WORDS_FILE, 'r', encoding='utf-8') as f:
            stop_words = set(f.read().splitlines())
    except FileNotFoundError:
        print(f"File {STOP_WORDS_FILE} not found. Please ensure the file exists.")
        exit(1)

# === Tiền xử lý và tách từ ===
final_sentences = []
for file in listOfFiles:
    sentences = []
    with open(file, 'r', encoding='utf-8') as f:
        text = f.readlines() 
        sentences.extend(text)

    # === Làm sạch câu ===
    for i in range(len(sentences)):
        sentence = sentences[i].strip()
        sentence = sentence.lower() 
        if REMOVE_PUNCTUATION:
            sentence = re.sub(r'[^\w\s\u00C0-\u1EF9]', '', sentence, flags=re.UNICODE)
        if REMOVE_STOP_WORDS:
            sentence = ' '.join([word for word in sentence.split() if word not in stop_words])
        sentences[i] = sentence

    # === Tách từ dùng underthesea ===
    for sentence in sentences:
        words = underthesea.word_tokenize(sentence)
        final_sentences.append(words)

# === Huấn luyện BM25 ===
model = BM25Okapi(final_sentences, b = 0, k = 1.2)

# Tạo thư mục output nếu chưa tồn tại
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

try:
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)

    # Trả về kết quả thành công
    result = {
        "message": "Training complete. BM25 saved.",
        "status": "success"
    }
except Exception as e:
    # Trả về thông báo lỗi nếu có lỗi xảy ra
    result = {
        "error": "Failed to save BM25 model",
        "details": str(e),
        "status": "fail"
    }

# In kết quả dưới dạng JSON
print(json.dumps(result, ensure_ascii=False, indent=4))
