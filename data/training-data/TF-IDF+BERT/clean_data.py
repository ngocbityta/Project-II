import pandas as pd

# Load dữ liệu
file_path = '../../raw-data/news.xlsx'
df = pd.read_excel(file_path)

# Loại bỏ các tiêu đề trùng lặp, chỉ giữ lại tiêu đề đầu tiên
df = df.drop_duplicates(subset=['title'], keep='first')

# Lưu lại vào chính file đã đọc dữ liệu
df.to_excel(file_path, index=False)

print("Data has been cleaned and saved back to the file.")