import streamlit as st
import pandas as pd
import joblib
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import re

import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

nltk.download('stopwords')
model = joblib.load('decision_tree_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')  


def preprocess_text(text):
    text = text.lower()  
    text = re.sub(r'[^\w\s]', '', text)  
    stop = stopwords.words('english')
    text = " ".join(x for x in text.split() if x not in stop) 
    return text

# Streamlit app
def main():
    st.title('Email Classification App')
    email_content = st.text_area("Paste the Email Content Here", height=300)
    
    if st.button('Predict'):
        if email_content:
            preprocessed_text = preprocess_text(email_content)
            vectorized_input = vectorizer.transform([preprocessed_text])
            prediction = model.predict(vectorized_input)
            if not prediction:
                 st.markdown(f"<h1 style='color: green;'>The email is classified as not Phishing: {prediction[0]}</h1>", unsafe_allow_html=True)
            else:
                st.markdown(f"<h1 style='color: red;'>The email is classified as Phishing!: {prediction[0]}</h1>", unsafe_allow_html=True)  
        else:
            st.error("Please paste an email content to analyze.")

if __name__ == "__main__":
    main()
