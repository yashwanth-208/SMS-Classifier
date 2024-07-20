import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
import joblib

# Load the SMS dataset into a pandas DataFrame
data = pd.read_csv('spam.csv')  # Replace with the actual path to your dataset

# Split the data into features and target
X = data['text']
y = data['type']

# Feature extraction using TF-IDF
tfidf_vectorizer = TfidfVectorizer()
X_tfidf = tfidf_vectorizer.fit_transform(X)

# Build and train the classification model (e.g., Linear SVC)
model = LinearSVC()
model.fit(X_tfidf, y)

# Save the trained model and TF-IDF vectorizer to files
joblib.dump(model, 'trained_model.pkl')
joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.pkl')