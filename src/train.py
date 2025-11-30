import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib
from preprocess import preprocess_text

# Define project root relative to this script
# src/train.py -> parent is src -> parent is project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load_data(csv_path: str):
    # Sentiment140 dataset has no header
    # Columns: target (0=negative, 4=positive), ids, date, flag, user, text
    col_names = ["target", "ids", "date", "flag", "user", "text"]
    df = pd.read_csv(csv_path, encoding="latin-1", header=None, names=col_names)
    
    # Map target: 0 -> 0, 4 -> 1
    df["label"] = df["target"].map({0: 0, 4: 1})
    
    # Drop rows with missing text or label
    df = df.dropna(subset=["text", "label"])
    return df


def main():
    # 1. Load data
    csv_path = os.path.join(PROJECT_ROOT, "data", "raw", "reviews.csv")
    print(f"Loading data from: {csv_path}")
    df = load_data(csv_path)

    # 2. Preprocess
    print("Preprocessing data...")
    df["clean_text"] = df["text"].apply(preprocess_text)

    # 3. Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_text"], df["label"],
        test_size=0.2, random_state=42, stratify=df["label"]
    )

    # 4. Vectorizer
    print("Vectorizing text...")
    vectorizer = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2)
    )

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # 5. Model
    print("Training model...")
    model = LogisticRegression(max_iter=1000)

    model.fit(X_train_vec, y_train)
    
    # 6. Evaluation
    y_pred = model.predict(X_test_vec)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    # 7. Save vectorizer & model
    models_dir = os.path.join(PROJECT_ROOT, "models")
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(vectorizer, os.path.join(models_dir, "tfidf_vectorizer.joblib"))
    joblib.dump(model, os.path.join(models_dir, "logreg_sentiment.joblib"))

    print(f"Model and vectorizer saved in '{models_dir}'")


if __name__ == "__main__":
    main()