# app.py
import pandas as pd
import re
import string
import gradio as gr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import nltk

nltk.download("punkt")

# =============================
# Load & Preprocess Dataset
# =============================
df = pd.read_csv("news.csv")   # dataset must have "text" and "category"

stemmer = PorterStemmer()

def preprocess(text):
    text = str(text).lower()
    text = re.sub(f"[{string.punctuation}]", "", text)
    tokens = word_tokenize(text)
    tokens = [stemmer.stem(word) for word in tokens]
    return " ".join(tokens)

df["clean_text"] = df["text"].apply(preprocess)

# =============================
# Train Model (Naive Bayes)
# =============================
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df["clean_text"])
y = df["category"]

model = MultinomialNB()
model.fit(X, y)

# =============================
# Prediction Function
# =============================
def predict_news(text):
    clean_text = preprocess(text)
    vec = vectorizer.transform([clean_text])
    prediction = model.predict(vec)[0]
    return prediction

# =============================
# Gradio Interface
# =============================
demo = gr.Interface(
    fn=predict_news,
    inputs=gr.Textbox(lines=5, placeholder="Enter news article text here..."),
    outputs="text",
    title="📰 News Classification",
    description="Classify news articles into categories using Naive Bayes."
)

if __name__ == "__main__":
    demo.launch()
