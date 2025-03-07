from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import joblib
import pickle
import os
from testing_the_model import preprocess_text, extract_text_features, pad_sequences_custom, predict_single_text, load_model_components

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS to allow frontend requests

# Load model components
components = load_model_components()

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data.get("text", "")

    if not text:
        return jsonify({"error": "No text provided"}), 400

    prediction = predict_single_text(
        text, 
        model=components["model"], 
        tokenizer=components["tokenizer"],
        tfidf_vectorizer=components["tfidf_vectorizer"],
        svd=components["svd"],
        scaler=components["scaler"]
    )
    
    # Fix the categorization issue
    ai_models = ["ChatGPT", "Claude", "Gemini"]
    human_models = ["Human"]
    
    # Calculate new aggregates
    ai_total = sum([prediction[model] for model in ai_models if model in prediction])
    human_total = sum([prediction[model] for model in human_models if model in prediction])
    
    # Update the prediction with corrected aggregates
    prediction["AI-generated (aggregate)"] = ai_total
    prediction["Human-written (aggregate)"] = human_total

    return jsonify(prediction)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)