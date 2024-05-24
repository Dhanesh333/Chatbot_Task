import urllib.request
from bs4 import BeautifulSoup
import re

def extract_wikipedia_text(url):
    response = urllib.request.urlopen(url)
    html = response.read()
    soup = BeautifulSoup(html, 'html.parser')
    paragraphs = soup.find_all('p')

    text_data = ''
    for para in paragraphs:
        text_data += para.get_text()
    
    cleaned_text = clean_text(text_data)
    return cleaned_text

def clean_text(text):
    text = re.sub(r'\[\d+\]', '', text)  # Removing references
    text = re.sub(r'\s+', ' ', text)  # Removing extra spaces
    return text.strip()

if __name__ == "__main__":
    url = 'https://en.wikipedia.org/wiki/Chatbot'
    text = extract_wikipedia_text(url)
    with open('cleaned_text.txt', 'w') as f:
        f.write(text)
