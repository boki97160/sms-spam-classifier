
import streamlit as st
import joblib
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Load the trained model and vectorizer
model = joblib.load('spam_classifier_model.joblib')
vectorizer = joblib.load('tfidf_vectorizer.joblib')

# Text preprocessing function
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # To lowercase
    text = text.lower()
    # Remove punctuation
    text = ''.join([char for char in text if char not in string.punctuation])
    # Tokenization
    tokens = nltk.word_tokenize(text)
    # Stopword removal and lemmatization
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Streamlit app
st.title('SMS Spam Classifier')
st.write('Enter a message to classify it as spam or ham.')

message_input = st.text_area('Message')

if st.button('Classify'):
    if message_input:
        processed_message = preprocess_text(message_input)
        vectorized_message = vectorizer.transform([processed_message])
        prediction = model.predict(vectorized_message)[0]
        st.write(f'**Prediction: {prediction.upper()}**')
    else:
        st.write('Please enter a message.')

st.header('Data Visualizations')

st.subheader('Class Distribution')
st.image('class_distribution.png')

st.subheader('Word Clouds')
st.image('ham_wordcloud.png')
st.image('spam_wordcloud.png')

st.subheader('Message Length Distribution')
st.image('message_length_distribution.png')
