import tensorflow as tf
import numpy as np
import pandas as pd
import pickle
import joblib
import re
import os

# Function to preprocess text
def preprocess_text(text):
    """Clean and normalize text for modeling"""
    if pd.isna(text):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Remove special characters and numbers (keeping spaces and basic punctuation)
    text = re.sub(r'[^\w\s.,!?]', '', text)
    
    # Replace multiple spaces with single space
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

# Function to calculate perplexity
def calculate_perplexity(text, n=3):
    """Calculate n-gram perplexity of text"""
    # Tokenize text
    tokens = text.lower().split()
    
    # Create n-grams
    ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
    
    if not ngrams:
        return 0
    
    # Count frequency of n-grams
    ngram_freq = {}
    for ngram in ngrams:
        if ngram in ngram_freq:
            ngram_freq[ngram] += 1
        else:
            ngram_freq[ngram] = 1
    
    # Calculate perplexity
    N = len(ngrams)
    if N == 0:
        return 0
    
    # Calculate entropy
    entropy = 0
    for ngram in ngram_freq:
        p = ngram_freq[ngram] / N
        entropy -= p * np.log2(p)
    
    # Calculate perplexity
    perplexity = 2 ** entropy
    return perplexity

# Function to extract text features
def extract_text_features(text):
    """Extract various statistical features from text"""
    if pd.isna(text) or text == "":
        return [0] * 11
    
    # Basic text statistics
    char_count = len(text)
    word_count = len(text.split())
    sentence_count = text.count('.') + text.count('!') + text.count('?')
    sentence_count = max(1, sentence_count)  # Avoid division by zero
    
    # Calculate averages
    avg_word_length = char_count / max(1, word_count)
    avg_sentence_length = word_count / sentence_count
    
    # Calculate special character ratios
    special_char_count = sum(1 for c in text if not c.isalnum() and not c.isspace())
    special_char_ratio = special_char_count / max(1, char_count)
    
    # Calculate uppercase ratio
    uppercase_count = sum(1 for c in text if c.isupper())
    uppercase_ratio = uppercase_count / max(1, char_count)
    
    # Calculate punctuation counts
    punctuation_count = sum(1 for c in text if c in ".,;:!?'\"()-")
    punctuation_ratio = punctuation_count / max(1, char_count)
    
    # Calculate perplexity measures
    perplexity_2gram = calculate_perplexity(text, n=2)
    perplexity_3gram = calculate_perplexity(text, n=3)
    
    # Calculate type-token ratio (lexical diversity)
    unique_words = len(set(text.lower().split()))
    type_token_ratio = unique_words / max(1, word_count)
    
    return [
        char_count, word_count, sentence_count, 
        avg_word_length, avg_sentence_length, 
        special_char_ratio, uppercase_ratio, punctuation_ratio,
        perplexity_2gram, perplexity_3gram, type_token_ratio
    ]

# Function for padding sequences
def pad_sequences_custom(sequences, maxlen, padding='pre', value=0.0):
    """Pads sequences to the same length."""
    num_samples = len(sequences)
    
    # Create output array
    x = np.full((num_samples, maxlen), value)
    
    for idx, s in enumerate(sequences):
        if len(s) == 0:
            continue  # empty sequence
            
        if len(s) >= maxlen:
            # Truncate
            trunc = s[-maxlen:]    
            x[idx, :] = trunc
        else:
            # Pad
            x[idx, -len(s):] = s
                
    return x

# Modified prediction function to handle various output formats
# Modified prediction function to handle 4-class output
def predict_single_text(text, model=None, tokenizer=None, max_sequence_length=200, 
                       tfidf_vectorizer=None, svd=None, scaler=None):
    """
    Predict if text is AI-generated or human-written with support for 4-class outputs
    """
    # If model is missing, we can't make predictions
    if model is None:
        return {"Error": "Model not loaded"}
        
    # Preprocess text
    cleaned_text = preprocess_text(text)
    
    # Extract text features
    stats_features = np.array([extract_text_features(cleaned_text)])
    
    # Apply scaler to stats features if available
    if scaler is not None:
        try:
            stats_features = scaler.transform(stats_features)
        except Exception as e:
            print(f"Warning: Could not scale features: {e}")
    
    # Process text with TF-IDF if available
    if tfidf_vectorizer is not None and svd is not None:
        try:
            tfidf_features = tfidf_vectorizer.transform([cleaned_text])
            tfidf_reduced = svd.transform(tfidf_features)
        except Exception as e:
            print(f"Warning: Could not process TF-IDF features: {e}")
            tfidf_reduced = np.zeros((1, 100))
    else:
        # Use zeros if vectorizer or SVD is missing
        tfidf_reduced = np.zeros((1, 100))
    
    # Process sequence data
    if tokenizer:
        sequences = tokenizer.texts_to_sequences([cleaned_text])
        padded_sequences = pad_sequences_custom(sequences, maxlen=max_sequence_length)
    else:
        # If tokenizer is missing, create empty sequences
        padded_sequences = np.zeros((1, max_sequence_length))
    
    # Make prediction with better error handling
    try:
        # Get raw prediction from model
        raw_prediction = model.predict([padded_sequences, tfidf_reduced, stats_features], verbose=0)
        
        # Debug information - print prediction shape
        print(f"DEBUG - Raw prediction shape: {raw_prediction.shape}")
        print(f"DEBUG - Raw prediction value: {raw_prediction}")
        
        # Handle different prediction shapes
        if len(raw_prediction.shape) == 1:
            # Single prediction value
            prediction = raw_prediction
        elif len(raw_prediction.shape) == 2:
            # Batch of predictions, take the first one
            prediction = raw_prediction[0]
        else:
            return {"Error": f"Unexpected prediction shape: {raw_prediction.shape}"}
        
        # Handle 4-class output (your model's current output)
        if len(prediction) == 4:
            # Define class names for the 4 classes
            class_names = ["ChatGPT", "Claude", "Gemini", "Human"]
            
            # Create result dictionary with percentages
            result = {}
            for i, class_name in enumerate(class_names):
                result[class_name] = float(prediction[i]) * 100
                
            # Add an aggregate AI vs. Human score if needed
            # Assuming classes 0 and 1 represent AI and classes 2 and 3 represent Human (adjust as needed)
            ai_score = prediction[0] + prediction[1]
            human_score = prediction[2] + prediction[3]
            
            result["AI-generated (aggregate)"] = float(ai_score) * 100
            result["Human-written (aggregate)"] = float(human_score) * 100
            
            return result
                
        # Handle single output value (binary classification with sigmoid)
        elif len(prediction) == 1:
            p = float(prediction[0])
            return {
                "AI-generated": p * 100,
                "Human-written": (1 - p) * 100
            }
        # Handle two output values (binary classification with softmax)
        elif len(prediction) == 2:
            return {
                "AI-generated": float(prediction[0]) * 100,
                "Human-written": float(prediction[1]) * 100
            }
        else:
            return {"Error": f"Unexpected number of output classes: {len(prediction)}"}
            
    except Exception as e:
        return {"Error": f"Prediction failed: {str(e)}"}

# Function to load model components with error handling
def load_model_components():
    components = {
        "model": None,
        "tokenizer": None,
        "tfidf_vectorizer": None,
        "svd": None,
        "scaler": None,
        "label_encoder": None
    }
    
    # List of current files in directory
    current_files = os.listdir('.')
    print(f"Files in current directory: {current_files}")
    
    # Load model
    if 'best_ai_detector_model.h5' in current_files:
        try:
            print("Loading model...")
            components["model"] = tf.keras.models.load_model('best_ai_detector_model.h5')
            print("Model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {e}")
    else:
        print("Model file 'best_ai_detector_model.h5' not found")
    
    # Load tokenizer
    if 'tokenizer.pickle' in current_files:
        try:
            print("Loading tokenizer...")
            with open('tokenizer.pickle', 'rb') as handle:
                components["tokenizer"] = pickle.load(handle)
            print("Tokenizer loaded successfully")
        except Exception as e:
            print(f"Error loading tokenizer: {e}")
    else:
        print("Tokenizer file 'tokenizer.pickle' not found")
    
    # Only attempt to load other components if they exist
    optional_components = [
        ("tfidf_vectorizer", "tfidf_vectorizer.joblib"),
        ("svd", "svd_reducer.joblib"),
        ("scaler", "scaler.joblib"),
        ("label_encoder", "label_encoder.joblib")
    ]
    
    for component_name, filename in optional_components:
        if filename in current_files:
            try:
                print(f"Loading {component_name}...")
                components[component_name] = joblib.load(filename)
                print(f"{component_name} loaded successfully")
            except Exception as e:
                print(f"Error loading {component_name}: {e}")
        else:
            print(f"{filename} not found, will use simplified approach")
    
    return components

# Main execution
if __name__ == "__main__":
    print("AI Detector Test Script")
    print("----------------------")
    
    # Load available components
    components = load_model_components()
    
    # Check if we have the minimum required components
    if components["model"] is None:
        print("ERROR: Model not found. Cannot proceed without a model.")
        exit(1)
    
    # Test samples
    test_texts = [
        "The sunset painted the sky with hues of orange and purple, casting long shadows across the landscape.",
        "The neural network architecture consists of multiple layers including convolutional and recurrent components for effective feature extraction.",
        "In this study, we propose a novel approach to sentiment analysis that leverages transformer architectures and contextual embeddings to capture nuanced emotional expressions in text.",
        "Once upon a time, there was a small village nestled between two mountains. The villagers lived simple but happy lives.",
        "The implementation of the algorithm requires careful consideration of edge cases to ensure robustness in production environments."
    ]
    
    # Test with example texts
    print("\n--- Testing with example texts ---")
    for i, text in enumerate(test_texts):
        print(f"\nExample {i+1}:")
        print(f"Text: {text[:100]}..." if len(text) > 100 else f"Text: {text}")
        
        # Make prediction using all available components
        prediction = predict_single_text(
            text, 
            model=components["model"], 
            tokenizer=components["tokenizer"],
            tfidf_vectorizer=components["tfidf_vectorizer"],
            svd=components["svd"],
            scaler=components["scaler"]
        )
        
        # Display results
        print("Predictions:")
        if "Error" in prediction:
            print(f"  {prediction['Error']}")
        else:
            for source, prob in sorted(prediction.items(), key=lambda x: x[1], reverse=True):
                print(f"  {source}: {prob:.2f}%")
    
    # Allow user to test custom text
    print("\n--- Test your own text ---")
    print("Enter your text (or type 'exit' to quit):")
    
    try:
        while True:
            custom_text = input()
            
            if custom_text.lower() == 'exit':
                break
            
            # Make prediction with all available components
            prediction = predict_single_text(
                custom_text, 
                model=components["model"], 
                tokenizer=components["tokenizer"],
                tfidf_vectorizer=components["tfidf_vectorizer"],
                svd=components["svd"],
                scaler=components["scaler"]
            )
            
            # Display results
            print("\nPredictions:")
            if "Error" in prediction:
                print(f"  {prediction['Error']}")
            else:
                for source, prob in sorted(prediction.items(), key=lambda x: x[1], reverse=True):
                    print(f"  {source}: {prob:.2f}%")
            
            print("\nEnter another text (or type 'exit' to quit):")
    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        print("Exiting...")