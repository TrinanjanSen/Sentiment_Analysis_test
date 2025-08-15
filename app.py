import streamlit as st
import pandas as pd
import joblib
import re
from nltk.stem.porter import PorterStemmer

# Load model and vectorizer
model = joblib.load("sentiment_model.pkl")        # change to your filename
vectorizer = joblib.load("vectorizer.pkl")        # change to your filename

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
st.write("Enter a tweet and find out if it’s **Positive**, **Negative**, or **Neutral**.")

tweet = st.text_area("Tweet text:")

if st.button("Analyze Sentiment"):
    if tweet.strip():
        stemmed_tweet = stemming(tweet)
        vectorized_tweet = vectorizer.transform([stemmed_tweet])
        prediction = model.predict(vectorized_tweet)[0]
        st.subheader(f"Prediction: {prediction}")
    else:
        st.warning("Please enter some text.")
