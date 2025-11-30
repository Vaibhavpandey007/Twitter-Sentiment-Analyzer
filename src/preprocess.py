import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


STOPWORDS = set(stopwords.words("english"))

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www.\S+", " ", text)       
    text = re.sub(r"[^a-zA-Z']", " ", text)            
    text = re.sub(r"\s+", " ", text).strip()            
    return text

def remove_stopwords(text: str) -> str:
    tokens = text.split()
    filtered = [t for t in tokens if t not in STOPWORDS]
    return " ".join(filtered)

def preprocess_text(text: str) -> str:
    text = clean_text(text)
    text = remove_stopwords(text)
    return text
