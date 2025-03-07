# AI Text Detector - Setup Instructions

This document provides step-by-step instructions to set up and run the AI Text Detector application. The application consists of a Python backend server and an HTML/JavaScript frontend.

## System Requirements

- Python 3.8 or higher
- Web browser (Chrome, Firefox, Edge recommended)
- Internet connection (for CDN resources)

## Files Included

- `ai-detector-frontend.html`: Frontend interface
- `server.py`: Flask backend server
- `testing_the_model.py`: Helper functions for text processing and prediction
- Model files (in the `model` directory):
  - `model.h5`: The trained neural network model
  - `tokenizer.pickle`: Text tokenizer
  - `tfidf_vectorizer.joblib`: TF-IDF vectorizer
  - `svd.joblib`: SVD model for dimensionality reduction
  - `scaler.joblib`: Feature scaler

## Setup Instructions

### 1. Install Python

If Python is not already installed:
- Download and install Python 3.8 or higher from [python.org](https://www.python.org/downloads/)
- Ensure that Python is added to your PATH during installation

### 2. Set Up Python Environment

Open a terminal or command prompt and run the following commands:

```bash
# Create a virtual environment (recommended but optional)
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install required packages
pip install -r requirements.txt
```

### 3. Start the Backend Server

With the virtual environment activated, run:

```bash
python server.py
```

You should see output indicating that the Flask server is running at `http://0.0.0.0:5000/`.

### 4. Open the Frontend Interface

- Open the file `ai-detector-frontend.html` in your preferred web browser
- You can do this by double-clicking the file or right-clicking and selecting "Open with" your browser

### 5. Using the Application

1. Enter or paste text (minimum 50 characters) into the text area
2. Click "Analyze Text" to process the input
3. View the results showing the probability of the text being AI-generated or human-written
4. You can also upload a .txt file using the "Upload File" button

## Troubleshooting

If you encounter any issues:

### Frontend Cannot Connect to Backend

- Ensure the backend server is running (you should see Flask running in your terminal)
- Check that your browser allows local file access to make API requests
- If you see CORS errors in your browser console, verify that CORS is enabled in the Flask app

### Python Package Issues

- Verify all required packages are installed: `pip list`
- Try reinstalling requirements: `pip install -r requirements.txt --force-reinstall`

### Model Loading Issues

- Ensure all model files are in the correct location (by default, within a `model` directory)
- Check console output for specific error messages related to file paths

## Additional Notes

- The backend server runs in debug mode by default, which is fine for testing
- For production use, set `debug=False` in server.py
- The API endpoint is configured to run on port 5000. If you need to use a different port, change the port number in both server.py and update the apiBaseUrl in the frontend HTML file

If you have any questions or need further assistance, please contact me directly.
