import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
import pickle
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Load model and preprocessing components
model = tf.keras.models.load_model('best_ai_detector_model.h5')

with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

tfidf_vectorizer = joblib.load('tfidf_vectorizer.joblib')
svd = joblib.load('svd_reducer.joblib')
scaler = joblib.load('scaler.joblib')

# Label mapping
label_mapping = {
    0: "Human-Written",
    1: "ChatGPT",
    2: "Gemini",
    3: "Claude"
}

# Helper functions
def custom_pad_sequences(sequences, maxlen):
    """Custom implementation of pad_sequences"""
    padded = []
    for seq in sequences:
        if len(seq) > maxlen:
            padded.append(seq[:maxlen])
        else:
            padded.append(seq + [0] * (maxlen - len(seq)))
    return np.array(padded)

def preprocess_text(text):
    """Clean and normalize text"""
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^\w\s.,!?]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip().lower()

def extract_text_features(text):
    """Extract statistical features from text"""
    words = text.split()
    char_count = len(text)
    word_count = len(words)
    unique_words = len(set(words))
    
    avg_word_length = char_count / max(1, word_count)
    unique_word_ratio = unique_words / max(1, word_count)
    
    sentences = text.split('.')
    avg_sentence_length = word_count / max(1, len(sentences))
    
    return [
        char_count, 
        word_count, 
        len(sentences),
        avg_word_length,
        unique_word_ratio,
        avg_sentence_length,
        sum(1 for c in text if c.isupper()) / max(1, char_count),
        text.count('!') / max(1, word_count),
        text.count('?') / max(1, word_count),
        len(re.findall(r'\d', text)) / max(1, char_count),
        0
    ]

def predict_text(text, debias=False, debias_matrix=None):
    """Make a prediction with optional debiasing"""
    # Preprocess text
    cleaned_text = preprocess_text(text)
    
    # Extract features
    stats_features = np.array([extract_text_features(text)])
    stats_features_scaled = scaler.transform(stats_features)
    
    # TF-IDF processing
    tfidf_features = tfidf_vectorizer.transform([cleaned_text])
    tfidf_reduced = svd.transform(tfidf_features)
    
    # Text sequence processing
    sequences = tokenizer.texts_to_sequences([cleaned_text])
    padded_sequences = custom_pad_sequences(sequences, maxlen=200)
    
    # Make prediction
    predictions = []
    for _ in range(5):
        prediction = model.predict([padded_sequences, tfidf_reduced, stats_features_scaled], verbose=0)[0]
        predictions.append(prediction)
    
    avg_prediction = np.mean(predictions, axis=0)
    
    # Apply debiasing if requested
    if debias and debias_matrix is not None:
        corrected_prediction = avg_prediction * debias_matrix
    else:
        corrected_prediction = avg_prediction
    
    # Convert to percentages
    result = {}
    total = sum(corrected_prediction)
    for i, label in label_mapping.items():
        result[label] = (corrected_prediction[i] / total) * 100
    
    return result, avg_prediction