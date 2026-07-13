# 📰 Fake News Detection using Machine Learning

![Python](https://img.shields.io/badge/Python-3.12+-blue)
![Flask](https://img.shields.io/badge/Flask-Web%20Application-black)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Logistic%20Regression-orange)
![NLP](https://img.shields.io/badge/NLP-GloVe-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

A Machine Learning and Natural Language Processing (NLP) based web application that classifies news articles as **Real** or **Fake** using **pre-trained GloVe word embeddings** and a **Logistic Regression** classifier.

---

# Features

- Detects whether a news article is **Real** or **Fake**
- Uses **GloVe (Global Vectors for Word Representation)** embeddings
- Logistic Regression classifier
- Confidence score for every prediction
- Flask REST API backend
- Clean and responsive web interface
- Light/Dark mode
- Google News RSS integration
- Input validation and error handling

---

# Project Workflow

```
User Input
      │
      ▼
Text Preprocessing
      │
      ▼
GloVe Word Embeddings
      │
      ▼
Logistic Regression Model
      │
      ▼
Prediction + Confidence Score
      │
      ▼
Flask API
      │
      ▼
Web Interface
```

---

# Technologies Used

| Category | Technology |
|-----------|------------|
| Programming Language | Python |
| Machine Learning | Scikit-learn |
| NLP | GloVe Embeddings |
| Backend | Flask |
| Frontend | HTML, CSS, JavaScript |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Model Storage | Pickle / Joblib |

---

# Dataset

This project uses the **WELFake Dataset**, containing **72,134** labelled news articles.

Columns:

- title
- text
- label

Label Encoding

| Label | Meaning |
|--------|----------|
| 0 | Fake News |
| 1 | Real News |

The dataset is **not included** in this repository.

---

# Repository Structure

```
Fake-News-Detection/

│── train_glove_model.py
│── prediction_glove.py
│── server.py
│── index.html
│── requirements.txt
│── README.md
│── LICENSE

├── screenshots/
│     ├── Home Page.png
│     ├── About.png
│     ├── Prediction_true.png
│     ├── Prediction_fake.png
│     ├── Google News RSS integration.png
│     ├── confusion_matrix.png
│     ├── precision_recall_curve.png
```

---

# Installation

## Clone Repository

```bash
git clone https://github.com/AdityaT10/Fake-News-Detection-Using-Machine-Learning.git

cd AdityaT10
```

---

## Create Virtual Environment (Optional)

Windows

```bash
python -m venv venv

venv\Scripts\activate
```

Linux / macOS

```bash
python3 -m venv venv

source venv/bin/activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Required Downloads

Large files are not stored directly inside this repository.

Before running the application, download:

### 1. Trained Model

Download

```
glove_logreg_model.sav
```

from the **GitHub Releases** section and place it in the project root directory.

---

### 2. GloVe Embeddings

Download

```
glove.6B.50d.txt
```

from the official Stanford NLP website:

https://nlp.stanford.edu/projects/glove/

Extract the file and place it in the project root.

---

### 3. Dataset (Optional)

If you want to retrain the model, download the **WELFake Dataset** from Kaggle.

Place

```
WELFake_Dataset.csv
```

inside the project folder.

---

# Running the Application

Start the Flask server

```bash
python server.py
```

Open your browser

```
http://127.0.0.1:5000
```

Paste a news article or headline into the text box and click **Predict**.

The application will display

- Prediction (Real/Fake)
- Confidence Score

---

# Screenshots

## Home Page

<img width="1898" height="868" alt="Home Page" src="https://github.com/user-attachments/assets/670ebf2f-97cb-4e90-a518-b34c27c03eda" />


---

## Prediction Result (Real News)

<img width="1897" height="867" alt="Fake News Detection 1" src="https://github.com/user-attachments/assets/86085606-513b-4dda-a0af-da6db57151ac" />


---

## Prediction Result (Fake News)

<img width="1892" height="869" alt="Fake News Detection 2" src="https://github.com/user-attachments/assets/296da5db-beb6-4ae3-b05b-3ce30281944e" />


---

## Live News

<img width="1898" height="873" alt="Live News" src="https://github.com/user-attachments/assets/f1bb951c-0cc9-469c-bc92-6645f96dd91e" />


---


# License

This project is licensed under the **MIT License**.

See the **LICENSE** file for more information.

# Author
**Aditya Thakur**
