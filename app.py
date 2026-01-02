from flask import Flask, render_template, request, jsonify
import pickle
import re
import nltk
from nltk.corpus import stopwords

# Initialize Flask app
app = Flask(__name__)

# Load model and vectorizer
with open("model/sentiment_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("model/vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Download stopwords (safe even if already downloaded)
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

# Text cleaning function (SAME as training)
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    words = [w for w in text.split() if w not in stop_words]
    return " ".join(words)

# Home route
@app.route("/")
def home():
    return render_template("index.html")

# Prediction route (AJAX)
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    user_text = data.get("text", "")

    if user_text.strip() == "":
        return jsonify({"error": "Empty input"})

    # Clean & vectorize text
    cleaned_text = clean_text(user_text)
    vectorized_text = vectorizer.transform([cleaned_text])

    # Prediction
    prediction = model.predict(vectorized_text)[0]
    probability = model.predict_proba(vectorized_text)[0].max()

    if prediction == 1:
        sentiment = "Positive 😊"
    else:
        sentiment = "Negative 😞"

    return jsonify({
        "sentiment": sentiment,
        "confidence": round(probability * 100, 2)
    })

# Run app
if __name__ == "__main__":
    app.run(debug=True)
