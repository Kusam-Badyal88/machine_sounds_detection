import os
import numpy as np
import librosa
import pickle
import logging
from flask import Flask, request, render_template, redirect, jsonify

# Set up logging
logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)

# Load the classifiers
model_paths = {
    "svm": "svm_model.pkl",
    "random_forest": "rf_model.pkl",
    "decision_tree": "dt_model.pkl",
    "naive_bayes": "nb_model.pkl"
}
classifiers = {}

for model_name, model_path in model_paths.items():
    try:
        with open(model_path, 'rb') as f:
            classifiers[model_name] = pickle.load(f)
            logging.debug(f"Loaded {model_name} model from {model_path}")
    except Exception as e:
        logging.error(f"Error loading {model_name} model from {model_path}: {e}")

# Accuracy values for each model
model_accuracies = {
    "svm": 0.89,
    "random_forest": 0.94,
    "decision_tree": 0.80,
    "naive_bayes": 0.57
}

# Define a mapping from string labels to integer labels
label_mapping = {
    'LM_NON-DEF': 0,
    'LM_DEF': 1,
    'VMC_NON-DEF': 2,
    'VMC_DEF': 3
}

# Function to extract features from audio files
def extract_features(file_path):
    try:
        audio, _ = librosa.load(file_path, sr=None)  # Load audio file

        # Extract Mel spectrogram
        mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=44100)

        # Extract power spectral density (PSD)
        psd = librosa.feature.rms(y=audio)

        # Extract MFCCs
        mfccs = librosa.feature.mfcc(y=audio, sr=44100, n_mfcc=20)

        # Extract zero crossing rate (ZCR)
        zcr = librosa.feature.zero_crossing_rate(y=audio)[0]

        # Extract spectral centroid
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=44100)

        # Fix the lengths of extracted features
        mel_spectrogram = librosa.util.fix_length(mel_spectrogram, size=200)
        psd = librosa.util.fix_length(psd, size=200)
        mfccs = librosa.util.fix_length(mfccs, size=200)
        zcr = librosa.util.fix_length(zcr, size=200)
        spectral_centroid = librosa.util.fix_length(spectral_centroid, size=200)

        # Flatten the features
        mel_spectrogram_flat = mel_spectrogram.flatten()
        psd_flat = psd.flatten()
        mfccs_flat = mfccs.flatten()
        zcr_flat = zcr.flatten()
        spectral_centroid_flat = spectral_centroid.flatten()

        # Concatenate all features into a single feature vector
        features = np.concatenate((mel_spectrogram_flat, psd_flat, mfccs_flat, zcr_flat, spectral_centroid_flat))

        return features
    except Exception as e:
        logging.error(f"Error extracting features from {file_path}: {e}")
        raise

# Function to classify audio file and provide reason
def classify_audio(audio_file, classifier):
    try:
        features = extract_features(audio_file)
        prediction = classifier.predict([features])[0]
        logging.debug(f"Prediction raw value: {prediction}")
        return prediction
    except Exception as e:
        logging.error(f"Error classifying audio file {audio_file}: {e}")
        raise

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'audio' not in request.files or 'algorithm' not in request.form:
        logging.error("Audio file or algorithm not provided")
        return redirect(request.url)
    
    file = request.files['audio']
    algorithm = request.form['algorithm']
    
    if file.filename == '':
        logging.error("No file selected")
        return redirect(request.url)
    
    if file and algorithm in classifiers:
        try:
            upload_folder = "C:/Users/OM/Downloads/fault_detection/uploads"
            os.makedirs(upload_folder, exist_ok=True)  # Create the uploads folder if it doesn't exist
            file_path = os.path.join(upload_folder, file.filename)
            file.save(file_path)
            classifier = classifiers[algorithm]
            prediction_raw = classify_audio(file_path, classifier)
            prediction = label_mapping[prediction_raw]

            accuracy = model_accuracies[algorithm] * 100  # Convert to percentage
            os.remove(file_path)
            # Capitalize the prediction value
            prediction_str = [key for key, value in label_mapping.items() if value == prediction][0].capitalize()
            # Pass algorithm as model_name to result.html
            return render_template('result.html', model_name=algorithm, prediction=prediction_str, accuracy=accuracy)
        except Exception as e:
            logging.error(f"Error during prediction: {e}")
            return render_template('error.html', error=str(e))
    
    return redirect(request.url)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        data = request.get_json()
        algorithm = data.get('algorithm')
        features = np.array(data.get('features')).reshape(1, -1)

        if algorithm not in classifiers:
            return jsonify({'error': 'Invalid algorithm specified'}), 400

        classifier = classifiers[algorithm]

        svm_pred = svm_model.predict(features)[0]
        rf_pred = rf_model.predict(features)[0]
        dt_pred = dt_model.predict(features)[0]
        nb_pred = nb_model.predict(features)[0]

        # Map predictions to integers
        svm_pred_int = label_mapping[svm_pred]
        rf_pred_int = label_mapping[rf_pred]
        dt_pred_int = label_mapping[dt_pred]
        nb_pred_int = label_mapping[nb_pred]

        # Verify predictions
        predictions_int = [svm_pred_int, rf_pred_int, dt_pred_int, nb_pred_int]
        are_predictions_valid = verify_predictions(predictions_int)

        if not are_predictions_valid:
            return jsonify({'error': 'Invalid predictions'}), 400

        return jsonify({
            'SVM Prediction': svm_pred_int,
            'Random Forest Prediction': rf_pred_int,
            'Decision Tree Prediction': dt_pred_int,
            'Naive Bayes Prediction': nb_pred_int,
            'Are predictions valid': are_predictions_valid
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def verify_predictions(predictions):
    for pred in predictions:
        if pred not in [0, 1, 2, 3]:
            return False
    return True

if __name__ == '__main__':
    app.run(debug=True)
