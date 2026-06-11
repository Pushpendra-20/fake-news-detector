from flask import Flask, render_template, request, jsonify
import pickle
import re
import os
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('stopwords', quiet=True)

app = Flask(__name__)

model = None
vectorizer = None

def load_model():
    global model, vectorizer
    try:
        with open('model/model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('model/vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Model not found: {e}")

def clean_text(text):
    ps = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
    tokens = text.split()
    tokens = [ps.stem(w) for w in tokens if w not in stop_words]
    return ' '.join(tokens)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data.get('text', '').strip()

    if not text:
        return jsonify({'error': 'No text provided'}), 400

    if model is None or vectorizer is None:
        return jsonify({'error': 'Model not loaded. Run train.py first.'}), 500

    cleaned = clean_text(text)
    features = vectorizer.transform([cleaned])
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0]

    confidence = round(float(max(probability)) * 100, 2)
    label = 'FAKE' if prediction == 1 else 'REAL'

    return jsonify({
        'label': label,
        'confidence': confidence,
        'fake_prob': round(float(probability[1]) * 100, 2),
        'real_prob': round(float(probability[0]) * 100, 2)
    })

if __name__ == '__main__':
    load_model()
    app.run(debug=True)
