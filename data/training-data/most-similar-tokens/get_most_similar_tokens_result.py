import json
import os
import sys
from underthesea import word_tokenize

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Hàm tách từ và chuẩn hóa
def tokenize(sentence):
    return set(word_tokenize(sentence.lower(), format='text').split())

if __name__ == "__main__":
    
    if len(sys.argv) < 2:
        print(json.dumps({"error": "No sentence provided."}))
        sys.exit(1)

    sentence = sys.argv[1]
    input_tokens = tokenize(sentence)
    
    try:
        news_file_path = os.path.join(CURRENT_DIR, '../../raw-data/news.txt')
        try:
            with open(news_file_path, 'r', encoding='utf-8') as file:
                sentences = file.readlines()
        except FileNotFoundError:
            print(json.dumps({"error": "Không tìm thấy tệp tin news.txt"}))
            sys.exit(1)

        similarities = []

        for news_sentence in sentences:
            news_sentence = news_sentence.strip()
            news_tokens = tokenize(news_sentence)
            common_tokens = input_tokens & news_tokens
            similarities.append({
                "cosine_similarity": len(common_tokens),
                "sentence": news_sentence
            })

        # Sắp xếp các câu theo số token giống nhau nhiều nhất
        similarities.sort(reverse=True, key=lambda x: x["cosine_similarity"])
        top_similarities = similarities[:10]

        print(json.dumps({"similarities": top_similarities}))

    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)
