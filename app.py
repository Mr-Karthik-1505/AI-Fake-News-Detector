from flask import Flask, render_template, request
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import os
print("Current Working Directory:", os.getcwd())

app = Flask(__name__)

# Load BERT model
model = BertForSequenceClassification.from_pretrained("bert_fake_news_model")
tokenizer = BertTokenizer.from_pretrained("bert_fake_news_model")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

from textblob import TextBlob

def analyze_bias(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity

    if polarity > 0.5:
        tone = "Strong Positive Tone"
    elif polarity < -0.5:
        tone = "Strong Negative Tone"
    elif polarity > 0:
        tone = "Slightly Positive"
    elif polarity < 0:
        tone = "Slightly Negative"
    else:
        tone = "Neutral"

    sensational_words = [
        "shocking", "unbelievable", "breaking",
        "secret", "exposed", "urgent", "massive"
    ]

    sensational_count = sum(word in text.lower() for word in sensational_words)

    return {
        "Tone": tone,
        "Sentiment Score": round(polarity, 3),
        "Sensational Word Count": sensational_count
    }

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    confidence = None
    bias_info = None

    if request.method == "POST":
        text = request.form["news"]

        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        )

        inputs = {key: val.to(device) for key, val in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=1)
            pred = torch.argmax(probs, dim=1).item()

        prediction = "Fake" if pred == 1 else "Real"
        confidence = round(probs[0][pred].item(), 4)
        bias_info = analyze_bias(text)

    # 👇 IMPORTANT: This must be OUTSIDE the if block
    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        bias=bias_info
    )

if __name__ == "__main__":
    app.run(debug=True)