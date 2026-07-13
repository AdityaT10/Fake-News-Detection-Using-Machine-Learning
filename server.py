"""
this code establishes a Flask web application that provides a user-friendly interface for detecting fake news. 
It loads the trained Logistic Regression model with GloVe embeddings to classify news articles 
as real or fake and offers an API endpoint for RSS feed headlines.
"""
from flask import Flask, request, jsonify, send_from_directory
import joblib
import feedparser

app = Flask(__name__, static_url_path='', static_folder='')

# loading the trained and saved model
clf, embedding_index = joblib.load("glove_logreg_model.sav")

def clean_text(text):
    import re
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def sentence_to_vec(sentence, embedding_index, dim=50):
    import numpy as np
    words = sentence.split()
    vectors = [embedding_index[w] for w in words if w in embedding_index]
    return np.mean(vectors, axis=0) if vectors else np.zeros(dim)

@app.route("/")
def serve_ui():
    return send_from_directory("", "index.html")


@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.json
    text = data.get("text", "")

    if not text:
        return jsonify({"error": "There is no text provided"}), 400

    # the user input get preprocess and vectorize
    cleaned = clean_text(text)
    vec = sentence_to_vec(cleaned, embedding_index)

    pred = clf.predict([vec])[0]
    proba = clf.predict_proba([vec])[0]

    # confidence score is generated
    try:
        confidence = float(max(proba))
    except:
        confidence = 0.0

    return jsonify({
        "label": "real" if pred == 1 else "fake",
        "confidence": confidence
    })

@app.route("/api/rss")
def rss():
    url = "https://news.google.com/rss"
    feed = feedparser.parse(url)
    headlines = [entry.title for entry in feed.entries[:15]]
    return jsonify(headlines)

if __name__ == "__main__":
    app.run(debug=True)