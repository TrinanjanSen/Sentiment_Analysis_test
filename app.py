import streamlit as st
import pandas as pd
import joblib
import re
from nltk.stem.porter import PorterStemmer

# Load model and vectorizer
import os
BASE_DIR = os.path.dirname(__file__)
model = joblib.load(os.path.join(BASE_DIR, "model.pkl"))
       # change to your filename
vectorizer = joblib.load(os.path.join(BASE_DIR, "vectorizer.pkl"))
      # change to your filename

# Stemmer setup
ps = PorterStemmer()

def stemming(content):
    content = re.sub('[^a-zA-Z]', ' ', content)
    content = content.lower()
    content = content.split()
    content = [ps.stem(word) for word in content]
    return ' '.join(content)

# Streamlit UI
st.title("Twitter Sentiment Analysis")
st.write("Enter a tweet and find out if it’s **Positive** or **Negative**.")

tweet = st.text_area("Tweet text:")

if st.button("Predict"):
    if tweet.strip():
        X = vectorizer.transform([tweet])
        prediction = model.predict(X)[0]  # 0 or 1

        # Map to labels
        if prediction == 0:
            sentiment = "Negative 😡"
        elif prediction == 1:
            sentiment = "Positive 😊"
        else:
            sentiment = "Unknown"

        st.success(f"Prediction: {sentiment}")
    else:
        st.warning("Please enter a tweet.")




