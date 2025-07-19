# 🎧 **Machine Sound Detection System** 🔍
An intelligent audio classification web application that detects defective and non-defective machine sounds using machine learning.
This project helps in early fault detection of machines using their sound signals, aiding in predictive maintenance for industrial applications.

It’s valuable for engineers, manufacturers, and technicians to monitor machine health and prevent breakdowns using audio diagnostics.

🚀 Key Features
🛠️ Real-Time Sound Analysis: Detects whether a machine is defective or non-defective.

🎙️ Audio Classification: Uses ML algorithms to classify uploaded machine sounds.

🧠 Multi-Model Support: Includes SVM, Random Forest, Decision Tree, and Naive Bayes.

📊 Accuracy Displayed: Each prediction includes the respective model's accuracy.

📁 Web Interface: Designed using Flask for a smooth user experience.

🎵 Feature Extraction: Uses MFCC, Chroma, and Spectral Centroid with Librosa.

🧠 Machine Learning Pipeline
Audio Input → .wav files
Preprocessing → Silence removal, normalization
Feature Extraction → MFCC, Chroma, Spectral Centroid
Model Training → SVM, RF, DT, NB using scikit-learn
Prediction → Flask interface provides instant classification

🖼️ Demo Screenshots
Upload Audio Page	Prediction Page	Model Selection
		

🔍 How It Works
🔊 User uploads a .wav audio file of machine noise.

📈 The app extracts features like:

MFCC (Mel Frequency Cepstral Coefficients)

Chroma Frequencies

Spectral Centroid

🧠 The chosen ML model predicts whether the machine is Defective or Non-Defective.

✅ The prediction result is shown along with model accuracy.

🧠 Model Predictions
Label	Description
0	LM_DEF
1	LM_NON-DEF
2	VMC_DEF
3	VMC_NON-DEF

You can map these internally for simpler outputs like "Defective" / "Non-Defective".

🛠️ Tech Stack
Tool / Library	Purpose
Python 🐍	Core Programming
Flask 🌐	Web Application Framework
Librosa 🎵	Audio Feature Extraction
NumPy 📈	Numerical Operations
Scikit-learn 🤖	Machine Learning Models
HTML/CSS	Frontend UI
Pickle 🧪	Save & Load ML Models

⚙️ How to Run the Project
bash
Copy
Edit
# Step 1: Clone the repo
git clone https://github.com/Kusam-Badyal88/machine_sounds_detection.git

# Step 2: Navigate to folder
cd machine_sounds_detection

# Step 3: Install all dependencies
pip install -r requirements.txt

# Step 4: Start the Flask server
python app.py

# Go to browser
http://127.0.0.1:5000

📦 Large File Downloads (Google Drive Links)
Due to GitHub's 100MB file size limit, please download large files manually:

🔹 machine_sounds_detection.ipynb
🔹 svm_model.pkl

📂 Project Structure
bash
Copy
Edit
machine_sounds_detection/
├── static/
│   └── screenshots/
│       ├── homepage.png
│       ├── prediction.png
│       └── model_selection.png
├── templates/
│   ├── index.html
│   └── result.html
├── uploads/
├── app.py
├── svm_model.pkl
├── rf_model.pkl
├── dt_model.pkl
├── nb_model.pkl
├── machine_sounds_detection.ipynb
└── README.md
📈 Model Accuracies
Model	Accuracy
SVM 📊	87%
Random Forest 🌲	91%
Decision Tree 🌴	85%
Naive Bayes 🧠	82%

✨ Future Improvements
📢 Add microphone-based live sound detection

🧠 Integrate deep learning models (CNN, LSTM)

📱 Create Android/iOS mobile app

🔁 Add real-time auto-refresh dashboard for live monitoring
