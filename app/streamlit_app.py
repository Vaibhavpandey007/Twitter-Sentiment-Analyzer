import streamlit as st
import sys
import os

# Make src importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
try:
    from inference import predict_sentiment
except ImportError:
    from src.inference import predict_sentiment

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Sentiment Analyzer",
    page_icon="📝",
    layout="centered",
)

# --- CUSTOM CSS FOR STYLING ---
st.markdown("""
<style>
/* Background */
.main {
    background-color: #0E1117;
}

/* Header styling */
.title {
    text-align: center;
    font-size: 45px;
    font-weight: 800;
    background: -webkit-linear-gradient(90deg, #ff8c00, #e52e71);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: -10px;
}

/* Subtitle */
.subtitle {
    text-align: center;
    font-size: 18px;
    color: #D3D3D3;
    margin-bottom: 40px;
}

/* Textbox */
textarea {
    border-radius: 12px !important;
    padding: 16px !important;
    font-size: 16px !important;
}

/* Button */
.stButton>button {
    background-color: #e52e71;
    color: white;
    padding: 10px 25px;
    border-radius: 10px;
    font-size: 17px;
    border: none;
    width: 100%;
}
.stButton>button:hover {
    background-color: #ff8c00;
    color: black;
    transition: 0.3s ease;
}

/* Sentiment Card */
.result-box {
    padding: 20px;
    border-radius: 12px;
    margin-top: 20px;
    font-size: 20px;
    text-align: center;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# --- HEADER ---
st.markdown("<h1 class='title'>Sentiment Analyzer</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Analyze text using Machine Learning & NLP 🧠</p>", unsafe_allow_html=True)

# --- INPUT BOX ---
user_text = st.text_area("Enter your text:", height=170, placeholder="Type something here...")

# --- BUTTON ---
if st.button("Analyze Sentiment 🚀"):
    if user_text.strip() == "":
        st.warning("⚠ Please enter some text to analyze.")
    else:
        sentiment = predict_sentiment(user_text)
        
        # Sentiment UI card
        if sentiment == "Positive":
            st.markdown(
                "<div class='result-box' style='background-color:#0f5132; color:#d1e7dd;'>😊 Positive Sentiment</div>",
                unsafe_allow_html=True
            )
        elif sentiment == "Negative":
            st.markdown(
                "<div class='result-box' style='background-color:#58151c; color:#f8d7da;'>😞 Negative Sentiment</div>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                "<div class='result-box' style='background-color:#084298; color:#cfe2ff;'>😐 Neutral Sentiment</div>",
                unsafe_allow_html=True
            )
