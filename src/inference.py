import os
import joblib
try:
    from src.preprocess import preprocess_text
except ImportError:
    from preprocess import preprocess_text

# Define PROJECT_ROOT relative to this script
# src/inference.py -> parent is src -> parent is project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "tfidf_vectorizer.joblib")
MODEL_PATH = os.path.join(MODEL_DIR, "logreg_sentiment.joblib")

# Load models
vectorizer = joblib.load(VECTORIZER_PATH)
model = joblib.load(MODEL_PATH)

def predict_sentiment(text: str) -> str:
    clean = preprocess_text(text)
    vec = vectorizer.transform([clean])
    pred = model.predict(vec)[0]
    # Map 0 -> Negative, 1 -> Positive
    return "Positive" if pred == 1 else "Negative"

if __name__ == "__main__":
    sample = "The product quality is amazing and I love it!"
    print(sample, "->", predict_sentiment(sample))