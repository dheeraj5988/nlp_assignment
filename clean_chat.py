import re
import emoji
from textblob import TextBlob
import pandas as pd
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

def normalize_emoji(text):
    return emoji.demojize(text)

def correct_spelling(text):
    return str(TextBlob(text).correct())

def clean_text(text):
    text = normalize_emoji(text.lower())
    text = re.sub(r"http\S+|www.\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = " ".join([word for word in text.split() if word not in stop_words])
    text = correct_spelling(text)
    return text

def process_chat(file_path):
    df = pd.read_csv(file_path)
    df['cleaned_message'] = df['message'].astype(str).apply(clean_text)
    df.to_csv('data/cleaned_chat.csv', index=False)
    print("Data cleaned and saved to data/cleaned_chat.csv")

# Example usage
# process_chat('data/healthygamer_gg_testdata.csv')
