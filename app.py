from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model and vectorizer
with open("model.pkl", "rb") as f:
    model = pickle.load(f)
with open("tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None  
    if request.method == "POST":
        news_title = request.form.get("news_title")  # Get title input
        news_text = request.form.get("news_text")    # Get article input

        if news_title and news_text:  
            full_text = news_title + " " + news_text  # Combine title + text
            full_text_vectorized = vectorizer.transform([full_text])  # Preprocess
            prediction = model.predict(full_text_vectorized)[0]  # Make prediction
            prediction = "True" if prediction == 1 else "Fake"  
    
    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)


