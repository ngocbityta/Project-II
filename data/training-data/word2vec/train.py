import json
import glob
import os
import re
import underthesea
from gensim.models import Word2Vec

# === Thiết lập cố định ===
INPUT_PATH = "../../raw-data/news.txt"
OUTPUT_PATH = "../../trained-data/vector.json"
REMOVE_STOP_WORDS = True
REMOVE_PUNCTUATION = True
STOP_WORDS_FILE = "../../raw-data/stopwords.txt"

# === Xử lý danh sách file ===
listOfFiles = [INPUT_PATH] if os.path.isfile(INPUT_PATH) else glob.glob(INPUT_PATH + '/*.txt')

# === Đọc stop words nếu có yêu cầu ===
stop_words = set()
if REMOVE_STOP_WORDS:
    try:
        with open(STOP_WORDS_FILE, 'r') as f:
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

# === Huấn luyện Word2Vec ===
model = Word2Vec(final_sentences, vector_size=100, window=5, min_count=5, workers=4)

# === Chuyển sang JSON và lưu ===
vectors = {"vectors": {word: model.wv[word].tolist() for word in model.wv.index_to_key}}

with open(OUTPUT_PATH, "w") as out_file:
    json.dump(vectors, out_file, indent = 2, ensure_ascii = False)

print(f"Training complete. Word vectors saved to {OUTPUT_PATH}")
