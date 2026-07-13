"""
The code is designed to train a Logistic Regression model that utilizes GloVe embeddings to spot fake news. 
It covers everything from loading and preprocessing the data to training the model, 
evaluating its performance, and saving the trained model along with its metrics.
"""
import numpy as np
import pandas as pd
import re
import pickle
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    precision_recall_curve, average_precision_score
)
import matplotlib.pyplot as plt
import seaborn as sns

# loading GloVe embeddings from file
def load_glove(path):
    print("Loading GloVe vectors...")
    embedding_index = {}
    with open(path, "r", encoding="utf8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype="float32")
            embedding_index[word] = vector
    print("Loaded GloVe words:", len(embedding_index))
    return embedding_index

# cleaning the text to remove unwanted characters, reduce the noise
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Converting the text to average GloVe vector
def sentence_to_vec(sentence, embedding_index, dim=50):
    words = sentence.split()
    vectors = [embedding_index[w] for w in words if w in embedding_index]
    return np.mean(vectors, axis=0) if vectors else np.zeros(dim)

# main function for the training and evaluation of the model
def main():
    print("Loading dataset...")
    df = pd.read_csv("WELFake_Dataset.csv")

    df["clean"] = df["text"].apply(clean_text)
    X = df["clean"]
    y = df["label"]

    glove_path = "glove.6B.50d.txt"
    embedding_index = load_glove(glove_path)

    print("Converting text to vectors...")
    X_vectors = np.vstack([sentence_to_vec(t, embedding_index) for t in tqdm(X)])

    X_train, X_test, y_train, y_test = train_test_split(
        X_vectors, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Training Logistic Regression...")
    clf = LogisticRegression(max_iter=2000)
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    probs = clf.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, preds)
    print("Accuracy:", acc)
    print(classification_report(y_test, preds))

    # Saving the trained model and embeddings
    with open("glove_logreg_model.sav", "wb") as f:
        pickle.dump((clf, embedding_index), f)

    print("Saved trained model.")

    # Confusion Matrix
    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix (Accuracy={acc:.2f})")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=300)
    plt.close()

    # Precision Recall Curve
    precision, recall, _ = precision_recall_curve(y_test, probs)
    ap = average_precision_score(y_test, probs)

    plt.figure(figsize=(6, 5))
    plt.step(recall, precision, where='post')
    plt.title(f"Precision-Recall Curve (AP={ap:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("precision_recall_curve.png", dpi=300)
    plt.close()

    # Saving the model report to text file
    with open("model_report.txt", "w") as f:
        f.write("Model: GloVe + Logistic Regression\n")
        f.write(f"Accuracy: {acc:.4f}\n\n")
        f.write("Classification Report:\n\n")
        f.write(classification_report(y_test, preds))

    print("Saved confusion_matrix.png, precision_recall_curve.png, model_report.txt")

if __name__ == "__main__":
    main()