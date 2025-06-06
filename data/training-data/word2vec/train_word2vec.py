import json
import glob
import os
import re
import underthesea
from gensim.models import Word2Vec

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_PATH = os.path.join(CURRENT_DIR, "../../raw-data/news.txt")
OUTPUT_PATH = os.path.join(CURRENT_DIR, "../../trained-data/word2vec/vector.json")
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
        
def normalize_sentence(s):
    # Loại bỏ ký tự xuống dòng, tab
    s = s.replace('\n', ' ').replace('\t', ' ')
    
    # Loại bỏ khoảng trắng đầu/cuối và chuyển về chữ thường
    s = s.strip().lower()
    
    # Loại bỏ toàn bộ ký tự không phải chữ cái, số hoặc khoảng trắng
    s = re.sub(r'[^\w\s]', '', s, flags=re.UNICODE)
    
    # Loại bỏ stop words
    s = ' '.join([word for word in s.split() if word not in stop_words])
    
    return s

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
            sentence = normalize_sentence(sentence)
        if REMOVE_STOP_WORDS:
            sentence = ' '.join([word for word in sentence.split() if word not in stop_words])
        sentences[i] = sentence

    # === Tách từ dùng underthesea ===
    for sentence in sentences:
        words = underthesea.word_tokenize(sentence)
        final_sentences.append(words)

# === Huấn luyện Word2Vec ===
model = Word2Vec(final_sentences, vector_size=50, window=4, min_count = 1, workers=4, sg=1, epochs=30)

# === Chuyển sang JSON và lưu ===
vectors = {"vectors": {word: model.wv[word].tolist() for word in model.wv.index_to_key}}

try:
    with open(OUTPUT_PATH, "w", encoding='utf-8') as out_file:
        json.dump(vectors, out_file, indent=2, ensure_ascii=False)

    # Trả về kết quả thành công
    result = {
        "message": "Training complete. Word vectors saved.",
        "status": "success"
    }
except Exception as e:
    # Trả về thông báo lỗi nếu có lỗi xảy ra
    result = {
        "error": "Failed to save Word2Vec vectors",
        "details": str(e),
        "status": "fail"
    }

# In kết quả dưới dạng JSON
print(json.dumps(result, ensure_ascii=False, indent=4))
