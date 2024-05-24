import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

def split_text_into_sentences(text):
    sentences = sent_tokenize(text)
    return sentences

if __name__ == "__main__":
    with open('cleaned_text.txt', 'r') as f:
        text = f.read()
    sentences = split_text_into_sentences(text)
    with open('sentences.txt', 'w') as f:
        for sentence in sentences:
            f.write("%s\n" % sentence)
