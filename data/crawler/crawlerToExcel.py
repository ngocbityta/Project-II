import openpyxl
import os
import time
import random
import pandas as pd
import sys
import json
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup

def crawNewsData(scroll_times):
    user_agents = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36"
    ]

    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--disable-infobars")
    options.add_argument("--disable-extensions")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument(f"user-agent={random.choice(user_agents)}")
    
    prefs = {"profile.managed_default_content_settings.images": 2}
    options.add_experimental_option("prefs", prefs)

    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)

    url = "https://lite.baomoi.com/"
    driver.get(url)

    def fast_scroll(driver, times):
        for i in range(times):
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(0.8)

    fast_scroll(driver, times=scroll_times)
    time.sleep(2)

    soup = BeautifulSoup(driver.page_source, "html.parser")
    driver.quit()

    articles = soup.find_all("a", class_="bm-card flex w-full md:flex-col")
    data = []

    for article in articles:
        title = article.attrs.get("title", "No title")
        img_tag = article.find("img")
        image = img_tag["src"] if img_tag else "No image"
        link = "https://lite.baomoi.com" + article.attrs["href"]
        data.append({"title": title, "image": image, "link": link})

    return data


def save_to_excel(data):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(BASE_DIR, "../raw-data/news.xlsx")

    df = pd.DataFrame(data)
    df.to_excel(output_path, index=False, engine='openpyxl')

    wb = openpyxl.load_workbook(output_path)
    ws = wb.active

    column_widths = {"A": 100, "B": 100, "C": 100}
    for col, width in column_widths.items():
        ws.column_dimensions[col].width = width

    wb.save(output_path)
    wb.close()

if __name__ == "__main__":
    try:
        scroll_times = 100
        news_data = crawNewsData(scroll_times)

        save_to_excel(news_data)

        result = {
            "message": "Crawling completed and saved to Excel successfully.",
            "total_articles": len(news_data),
            "status": "success"
        }
    except Exception as e:
        result = {
            "error": "Failed to crawl and save news",
            "details": str(e),
            "status": "fail"
        }

    print(json.dumps(result, ensure_ascii=False, indent=2))
