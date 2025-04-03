import pandas as pd

# Đọc file Excel
df = pd.read_excel('news.xlsx')

titles = df['title']

# Ghi các tiêu đề vào file news.txt
with open('news.txt', 'w', encoding='utf-8') as file:
    for title in titles:
        file.write(title + '\n')

print("Đã ghi các tiêu đề vào news.txt")
