import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn import svm
import requests
import io
import string
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
import pickle

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

# Load the selection features from the pickle file
with open('selected_features.pickle', 'rb') as file:
    selected_features = pickle.load(file)

# Function to train the SVM model
def train_svm(X_train, y_train, test_size, kernel):
    vectorizer = TfidfVectorizer(tokenizer=preprocess_data)
    X_train = vectorizer.fit_transform(X_train)
    X_train = X_train[selected_features]
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=test_size, random_state=42)
    clf = svm.SVC(kernel=kernel)
    clf.fit(X_train, y_train)
    return clf, vectorizer

# Function to make sentiment predictions
def predict_sentiment(text, clf, vectorizer):
    text = vectorizer.transform([text])
    prediction = clf.predict(text)
    return prediction[0]

# Load the dataset from GitHub
data = pd.read_excel("https://raw.githubusercontent.com/febyfadlilah/dataset/main/DataCrawlingTwitter2019Label.xlsx")

# Split features and labels
X_train = data['Text']
y_train = data['Label']

# Muat parameter terbaik dari file pickle
with open('best_params.pkl', 'rb') as file:
    best_params = pickle.load(file)

best = best_params[0]

# Ambil parameter terbaik
best_test_size = best[0]
best_kernel = best[1]

# Train the SVM model using the best parameters
clf, vectorizer = train_svm(X_train, y_train, best_test_size, best_kernel)


# Streamlit app title
st.title('Analisis Sentimen Tweets Pilpres 2019')
st.image(image, caption='Analisis Sentimen Pilpres')

# Input text for prediction
input_text = st.text_input('Masukkan teks:', '')

# Button to perform sentiment prediction
if st.button('Prediksi Sentimen'):
    if input_text:
        prediction = predict_sentiment(input_text, clf, vectorizer)
        st.write('Prediksi sentimen:',prediction)
    else:
        st.write('Silakan masukkan teks untuk melakukan prediksi.')
