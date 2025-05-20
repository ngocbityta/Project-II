from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
import json
import glob
import os
import re
import underthesea

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_PATH = os.path.join(CURRENT_DIR, "../../raw-data/news.txt")
VECTORS_PATH = os.path.join(CURRENT_DIR, "../../trained-data/doc2vec/vector.json")
MODEL_PATH = os.path.join(CURRENT_DIR, "../../trained-data/doc2vec/doc2vec.model")
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
documents = []
raw_sentences = {}  # map tag -> raw sentence
for file in listOfFiles:
    with open(file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            sentence = line.strip().lower()
            if not sentence:
                continue
            if REMOVE_PUNCTUATION:
                sentence = re.sub(r'[^\w\s\u00C0-\u1EF9]', '', sentence, flags=re.UNICODE)
            words = underthesea.word_tokenize(sentence)
            if REMOVE_STOP_WORDS:
                words = [word for word in words if word not in stop_words]

            tag = f'doc_{len(documents)}'
            documents.append(TaggedDocument(words=words, tags=[tag]))
            raw_sentences[tag] = line.strip()  # lưu raw chưa lower

# === Huấn luyện mô hình Doc2Vec ===
model = Doc2Vec(
    documents=documents,
    vector_size=50,
    window=5,
    min_count=1,
    workers=3,
    hs=1,
    dm=0,  # DBOW
    negative=0,
    dbow_words=1,
    epochs=100
)

# === Lưu vector của các document ===
vectors = {
    "vectors": {
        raw_sentences[doc.tags[0]]: model.dv[doc.tags[0]].tolist()
        for doc in documents
    }
}

try:
    # Lưu vector
    with open(VECTORS_PATH, "w", encoding="utf-8") as out_file:
        json.dump(vectors, out_file, indent=2, ensure_ascii=False)

    # Lưu mô hình
    model.save(MODEL_PATH)

    result = {
        "message": "Training complete. Vectors and model saved.",
        "status": "success"
    }
except Exception as e:
    result = {
        "error": "Failed to save Doc2Vec output",
        "details": str(e),
        "status": "fail"
    }

print(json.dumps(result, ensure_ascii=False, indent=4))
