# Twitter Sentiment Analyzer

This project analyzes the sentiment of tweets or text (Positive, Negative, Neutral) using Machine Learning and Natural Language Processing (NLP). It includes a Streamlit web application for real-time predictions.

---

## Features
- Text preprocessing (cleaning, removing stopwords)
- TF-IDF vectorizer
- Logistic Regression sentiment classifier
- Real-time Streamlit web app
- Modular project structure (train, inference, UI)

---
<img width="1920" height="1080" alt="Screenshot (20)" src="https://github.com/user-attachments/assets/72cc5f29-868f-42dd-99ed-40ab0bc98894" />



## Project Structure
Twitter-Sentiment-Analyzer/
│
├── app/
│ └── streamlit_app.py
│
├── src/
│ ├── train.py
│ ├── inference.py
│ └── preprocess.py
│
├── models/
│ ├── tfidf_vectorizer.joblib
│ └── logreg_sentiment.joblib
│
├── requirements.txt
└── README.md

---

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/Vaibhavpandey007/Twitter-Sentiment-Analyzer.git
cd Twitter-Sentiment-Analyzer
2. Create a virtual environment (optional)
python -m venv venv
venv\Scripts\activate   # Windows
# or
source venv/bin/activate   # Mac/Linux
3. Install dependencies
pip install -r requirements.txt
Training the Model

If you want to retrain the model:

cd src
python train.py


This generates new model files in the models/ directory.

Run the Streamlit App
cd app
streamlit run streamlit_app.py


Open in browser:

http://localhost:8501

Example Predictions

“I love this new update!” → Positive

“This is the worst experience.” → Negative

“It's okay, not too bad.” → Neutral

Technologies Used

Python

NLTK

Scikit-Learn

Pandas

NumPy

Streamlit

Joblib

Author

Vaibhav Pandey
GitHub: Vaibhavpandey007

