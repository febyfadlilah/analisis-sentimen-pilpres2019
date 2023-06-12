import streamlit as st
import pandas as pd
# import requests
import io
import string
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
import pickle
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer

import altair as alt
from PIL import Image

image = Image.open('pilpres.jpg')

nltk.download('punkt')
nltk.download('stopwords')

def preprocess_data(text):
    # Case folding
    text = text.lower()
    text = re.sub("\n", " ", text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    text = re.sub(r'\d+', '', text)

    # Punctual removal
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Tokenization
    text = nltk.word_tokenize(text)

    # Removing stopwords
    stop_words = set(stopwords.words('indonesian'))
    factory = StopWordRemoverFactory()
    stopword_remover = factory.create_stop_word_remover()
    text = [token for token in text if token not in stop_words]
    text = [stopword_remover.remove(token) for token in text]

    # Stemming
    stemmer = PorterStemmer()
    text = [stemmer.stem(token) for token in text]

    return text

# Function to make sentiment predictions
def predict_sentiment(text, clf, vectorizer):
    text = vectorizer.transform([text])
    prediction = clf.predict(text)
    return prediction[0]

# Load the dataset from GitHub
data = pd.read_excel("https://raw.githubusercontent.com/febyfadlilah/dataset/main/DataCrawlingTwitter2019Label.xlsx")

# Load the selection features from the pickle file
with open('selected_features.pickle', 'rb') as file:
    selected_features = pickle.load(file)

# Split features and labels
X_train = data['Text']
y_train = data['Label']

# Transform the text data using TF-IDF with selected features
vectorizer = TfidfVectorizer(tokenizer=preprocess_data, vocabulary=selected_features)
X_train = vectorizer.fit_transform(X_train)


# Save the trained model to a pickle file
with open('svm_model.pickle', 'rb') as file:
    clf = pickle.load(file)

# Train the SVM model
clf.fit(X_train, y_train)


# Streamlit app title
st.title('Analisis Sentimen Tweets Pilpres 2019')
st.image(image, caption='Analisis Sentimen Pilpres')

# Input text for prediction
input_text = st.text_input('Masukkan teks:', '')

# Button to perform sentiment prediction
if st.button('Prediksi Sentimen'):
    if input_text:
        processed_text = preprocess_data(input_text)
        processed_text = ' '.join(processed_text)
        processed_text = vectorizer.transform([processed_text])
        prediction = clf.predict(processed_text)
        st.write('Prediksi sentimen:', prediction[0])
    else:
        st.write('Silakan masukkan teks untuk melakukan prediksi.')
