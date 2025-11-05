
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib

# --- NLTK Data Download ---
def download_nltk_data():
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet')

download_nltk_data()


# --- Helper Functions ---

def preprocess_text(text):
    """Preprocesses the text data."""
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

def train_model(data, text_column, label_column):
    """Trains a Logistic Regression model."""
    X_train, X_test, y_train, y_test = train_test_split(data[text_column], data[label_column], test_size=0.2, random_state=42)
    vectorizer = TfidfVectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_train)
    model = LogisticRegression()
    model.fit(X_train_tfidf, y_train)
    return model, vectorizer

# --- Streamlit App ---

st.title("Interactive Text Classifier and Visualizer")

# --- 1. Dataset Selection ---
st.header("1. Select or Upload Dataset")
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
use_default_dataset = st.checkbox("Use the default SMS spam dataset", value=True)

df = None
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, encoding='latin-1')
elif use_default_dataset:
    df = pd.read_csv('Hands-On-Artificial-Intelligence-for-Cybersecurity/Chapter03/datasets/sms_spam_no_header.csv', encoding='latin-1', header=None)
    df.columns = ['label', 'message']

if df is not None:
    st.subheader("Dataset Preview")
    st.write(df.head())

    # --- 2. Column Selection ---
    st.header("2. Select Columns")
    label_column = st.selectbox("Select the label column", df.columns)
    text_column = st.selectbox("Select the text column", df.columns)

    # --- 3. Processing and Visualization ---
    if st.button("Analyze and Visualize"):
        st.header("3. Analysis and Visualization")

        # --- Class Distribution ---
        st.subheader("Class Distribution")
        fig, ax = plt.subplots()
        sns.countplot(x=label_column, data=df, ax=ax)
        st.pyplot(fig)

        # --- Text Preprocessing ---
        with st.spinner("Preprocessing text data..."):
            df['processed_text'] = df[text_column].apply(preprocess_text)
        st.subheader("Preprocessed Text Preview")
        st.write(df[[text_column, 'processed_text']].head())

        # --- Word Clouds ---
        st.subheader("Word Clouds")
        labels = df[label_column].unique()
        for label in labels:
            st.write(f"Word Cloud for '{label}'")
            words = ' '.join(df[df[label_column] == label]['processed_text'])
            if words:
                wordcloud = WordCloud(width=800, height=400, background_color='white').generate(words)
                fig, ax = plt.subplots()
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis('off')
                st.pyplot(fig)
            else:
                st.write("Not enough words to generate a word cloud.")

        # --- Message Length Analysis ---
        st.subheader("Message Length Analysis")
        df['message_length'] = df[text_column].apply(len)
        fig, ax = plt.subplots()
        sns.histplot(data=df, x='message_length', hue=label_column, multiple='stack', bins=50, ax=ax)
        st.pyplot(fig)

        # --- 4. Model Training and Classification ---
        st.header("4. Spam Classification")
        with st.spinner("Training classification model..."):
            model, vectorizer = train_model(df, 'processed_text', label_column)
        st.success("Model trained successfully!")

        message_input = st.text_area("Enter a message to classify")
        if st.button("Classify Message"):
            if message_input:
                processed_message = preprocess_text(message_input)
                vectorized_message = vectorizer.transform([processed_message])
                prediction = model.predict(vectorized_message)[0]
                st.write(f"**Prediction: {prediction.upper()}**")
            else:
                st.write("Please enter a message.")
