import pandas as pd
import json
import os

def get_title():
    try:
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))

        # Thiết lập đường dẫn tuyệt đối tới file input và output
        input_path = os.path.join(BASE_DIR, "../raw-data/corpus-title.txt")
        output_path = os.path.join(BASE_DIR, "../raw-data/news.txt")

        with open(input_path, 'r', encoding='utf-8') as infile:
            lines = infile.readlines()

        titles = lines[:100000]

        with open(output_path, 'w', encoding='utf-8') as file:
            for title in titles:
                file.write(str(title))

        return {
            "message": "Đã ghi 100.000 tiêu đề vào news.txt",
            "status": "success"
        }

    except Exception as e:
        return {
            "error": "Lỗi khi ghi file",
            "details": str(e),
            "status": "fail"
        }

if __name__ == '__main__':
    result = get_title()
    print(json.dumps(result, ensure_ascii=False, indent=4))
