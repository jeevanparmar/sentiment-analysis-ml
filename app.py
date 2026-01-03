from flask import Flask, render_template, request, jsonify
import pickle
import re
import nltk
from nltk.corpus import stopwords

# Initialize Flask app
app = Flask(__name__)



with open("model/sentiment_pipeline.pkl", "rb") as f:
    model = pickle.load(f)
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
    # vectorized_text = vectorizer.transform([cleaned_text])

    
    prediction = model.predict([cleaned_text])[0]
    probability = model.predict_proba([cleaned_text]).max()


    if prediction == 1:
        sentiment = "Positive 😊"
    else:
        sentiment = "Negative 😞"

    return jsonify({
        "sentiment": sentiment,
        "confidence": round(probability * 100, 2)
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

