import pandas as pd
import pickle
import re
import os
import sys
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

nltk.download('stopwords', quiet=True)

def clean_text(text):
    ps = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    text = re.sub(r'[^a-zA-Z\s]', '', str(text).lower())
    tokens = text.split()
    tokens = [ps.stem(w) for w in tokens if w not in stop_words]
    return ' '.join(tokens)

def load_kaggle_data():
    fake = pd.read_csv('data/Fake.csv')
    real = pd.read_csv('data/True.csv')
    fake['label'] = 1
    real['label'] = 0
    df = pd.concat([fake, real], ignore_index=True)
    df['content'] = df['title'] + ' ' + df['text']
    return df[['content', 'label']]

def load_auto_data():
    try:
        from datasets import load_dataset
        print("Downloading dataset from HuggingFace...")
        dataset = load_dataset("GonzaloA/fake_news")
        train_data = dataset['train'].to_pandas()
        train_data = train_data.rename(columns={'text': 'content'})
        return train_data[['content', 'label']].dropna()
    except Exception as e:
        print(f"Auto download failed: {e}")
        sys.exit(1)

def train():
    os.makedirs('model', exist_ok=True)

    if '--auto' in sys.argv:
        df = load_auto_data()
    elif os.path.exists('data/Fake.csv') and os.path.exists('data/True.csv'):
        df = load_kaggle_data()
    else:
        print("No dataset found. Run: python train.py --auto")
        sys.exit(1)

    print(f"Dataset loaded: {len(df)} samples")
    print(f"Fake: {df['label'].sum()} | Real: {(df['label']==0).sum()}")

    print("Cleaning text...")
    df['clean'] = df['content'].apply(clean_text)

    print("Vectorizing...")
    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
    X = vectorizer.fit_transform(df['clean'])
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("Training model...")
    model = LogisticRegression(max_iter=1000, C=1.0)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {acc * 100:.2f}%")
    print(classification_report(y_test, y_pred, target_names=['Real', 'Fake']))

    with open('model/model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('model/vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)

    print("Model saved to /model folder!")

if __name__ == '__main__':
    train()
