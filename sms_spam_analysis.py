
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

# Load the dataset
df = pd.read_csv('Hands-On-Artificial-Intelligence-for-Cybersecurity/Chapter03/datasets/sms_spam_no_header.csv', encoding='latin-1', header=None)

# Add column names
df.columns = ['label', 'message']

# Print the first 5 rows
print("First 5 rows of the dataset:")
print(df.head())

# Print class distribution
print("\nClass distribution:")
print(df['label'].value_counts())

# Create a bar plot of the class distribution
plt.figure(figsize=(8, 6))
sns.countplot(x='label', data=df)
plt.title('Class Distribution (Ham vs. Spam)')
plt.xlabel('Label')
plt.ylabel('Count')
plt.savefig('class_distribution.png')

print("\nSaved class distribution plot to class_distribution.png")

# Text preprocessing
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

df['processed_message'] = df['message'].apply(preprocess_text)

print("\nFirst 5 rows with processed message:")
print(df.head())

# Word Clouds
ham_words = ' '.join(df[df['label'] == 'ham']['processed_message'])
spam_words = ' '.join(df[df['label'] == 'spam']['processed_message'])

ham_wordcloud = WordCloud(width=800, height=400, background_color='white').generate(ham_words)
plt.figure(figsize=(10, 5))
plt.imshow(ham_wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud for Ham Messages')
plt.savefig('ham_wordcloud.png')
print("\nSaved ham word cloud to ham_wordcloud.png")

spam_wordcloud = WordCloud(width=800, height=400, background_color='white').generate(spam_words)
plt.figure(figsize=(10, 5))
plt.imshow(spam_wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud for Spam Messages')
plt.savefig('spam_wordcloud.png')
print("\nSaved spam word cloud to spam_wordcloud.png")

# Message Length Analysis
df['message_length'] = df['message'].apply(len)

plt.figure(figsize=(12, 6))
sns.histplot(data=df, x='message_length', hue='label', multiple='stack', bins=50)
plt.title('Message Length Distribution (Ham vs. Spam)')
plt.xlabel('Message Length')
plt.ylabel('Count')
plt.savefig('message_length_distribution.png')
print("\nSaved message length distribution plot to message_length_distribution.png")

# Model Training
X_train, X_test, y_train, y_test = train_test_split(df['processed_message'], df['label'], test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)

model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

# Save the model and vectorizer
joblib.dump(model, 'spam_classifier_model.joblib')
joblib.dump(vectorizer, 'tfidf_vectorizer.joblib')

print("\nSaved model to spam_classifier_model.joblib")
print("Saved vectorizer to tfidf_vectorizer.joblib")
