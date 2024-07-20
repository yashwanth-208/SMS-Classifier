import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
import joblib

# Load the trained model and TF-IDF vectorizer
model = joblib.load('trained_model.pkl')  # Replace with the actual path to your trained model
tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')  # Replace with the actual path to your TF-IDF vectorizer

# Function to predict whether the SMS message is spam or ham
def predict_spam_or_ham(message):
    message_tfidf = tfidf_vectorizer.transform([message])
    prediction = model.predict(message_tfidf)
    return prediction[0]

def main():
    st.title('SMS Spam Detection')

    # Text input for user to enter a new SMS message
    user_input = st.text_input('Enter a new SMS message:')
    if st.button('Predict'):
        if user_input:
            prediction = predict_spam_or_ham(user_input)
            st.write(f'Prediction: {prediction}')

if __name__ == '__main__':
    main()