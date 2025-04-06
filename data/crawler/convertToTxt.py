import pandas as pd
import json
import os

def convert_to_txt():
    try:
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))

        # Thiết lập đường dẫn tuyệt đối tới file input và output
        excel_path = os.path.join(BASE_DIR, "../raw-data/news.xlsx")
        txt_path = os.path.join(BASE_DIR, "../raw-data/news.txt")

        df = pd.read_excel(excel_path)
        titles = df['title']

        with open(txt_path, 'w', encoding='utf-8') as file:
            for title in titles:
                file.write(title + '\n')

        return {
            "message": "Đã ghi các tiêu đề vào news.txt",
            "status": "success"
        }

    except Exception as e:
        return {
            "error": "Lỗi khi ghi file",
            "details": str(e),
            "status": "fail"
        }

if __name__ == '__main__':
    result = convert_to_txt()
    print(json.dumps(result, ensure_ascii=False, indent=4))
