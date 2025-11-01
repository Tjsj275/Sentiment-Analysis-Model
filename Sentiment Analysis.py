import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
import gradio as gr

# --- Load and Prepare Data ---
data = pd.read_csv('movie_review.csv')
X = data.iloc[:, -2]
Y = data.iloc[:, -1]

# --- Train-Test Split ---
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=1)

# --- Build and Train Model ---
text_clf = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', LinearSVC())
])
text_clf.fit(X_train, y_train)

# --- Evaluate ---
y_pred = text_clf.predict(X_test)
accuracy = accuracy_score(y_pred, y_test)

# --- Gradio Function ---
def predict_sentiment(statement):
    prediction = text_clf.predict([statement])[0]
    return f"Predicted Sentiment: {prediction}"

# --- Interface ---
interface = gr.Interface(
    fn=predict_sentiment,
    inputs=gr.Textbox(lines=3, placeholder="Type a movie review..."),
    outputs="text",
    title="🎬 Sentiment Analysis App",
    description=f"Predicts sentiment (positive/negative). Accuracy: {accuracy:.2f}"
)

if __name__ == "__main__":
    interface.launch()
