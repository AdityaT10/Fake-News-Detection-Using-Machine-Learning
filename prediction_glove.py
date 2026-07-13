"""
this code can load a pre-trained Logistic Regression model that uses GloVe embeddings for fake news detection. 
It features functions for cleaning text, converting that text into GloVe vectors, 
and making predictions based on what the user inputs.
"""
import pickle
import numpy as np
import re

# clean the text input by the user
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# convert the cleaned text into GloVe vector representation
def sentence_to_vec(sentence, embedding_index, dim=50):
    words = sentence.split()
    vectors = []

    for w in words:
        if w in embedding_index:
            vectors.append(embedding_index[w])

    if len(vectors) == 0:
        return np.zeros(dim)
    
    return np.mean(vectors, axis=0)

# load the trained model and embeddings
print("Loading model...")
clf, embedding_index = pickle.load(open("glove_logreg_model.sav", "rb"))
print("Model loaded successfully!")

# user enters text for prediction
while True:
    text = input("\nEnter news text (or 'exit' to quit):\n> ")

    if text.strip().lower() == "exit":
        break

    cleaned = clean_text(text)
    vec = sentence_to_vec(cleaned, embedding_index)

    pred = clf.predict([vec])[0]
    prob = clf.predict_proba([vec])[0]

    print("\n--- Prediction ---")
    if pred == 1:
        print("✅ REAL NEWS")
    else:
        print("❌ FAKE NEWS")

    print(f"Real probability: {prob[1]*100:.2f}%")
    print(f"Fake probability: {prob[0]*100:.2f}%")