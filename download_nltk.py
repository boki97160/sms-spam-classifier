
import nltk
import os

# Create a directory to store the NLTK data
if not os.path.exists('nltk_data'):
    os.makedirs('nltk_data')

# Set the NLTK data path to the local directory
nltk.data.path.append('./nltk_data')

# Download the necessary NLTK data
nltk.download('stopwords', download_dir='./nltk_data')
nltk.download('punkt', download_dir='./nltk_data')
nltk.download('wordnet', download_dir='./nltk_data')

print("NLTK data downloaded to the 'nltk_data' directory.")
