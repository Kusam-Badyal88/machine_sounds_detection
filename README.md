# 🎧 Machine Sound Detection System 🔍

An intelligent audio classification web app that detects **defective** and **non-defective** machine sounds using Machine Learning.  
This project helps identify early signs of machine faults through sound analysis, supporting predictive maintenance in industrial environments.

---

## 🌟 Key Features

🛠️ Real-time audio classification of machine sounds  
🤖 Supports Multiple ML Models: **SVM**, **Random Forest**, **Decision Tree**, **Naive Bayes**  
📊 Displays Prediction and Model Accuracy  
🌐 User-Friendly Web Interface using Flask  
🎵 Advanced Feature Extraction with **MFCC**, **Chroma**, and **Spectral Centroid**  
📁 Upload Machine Audio (`.wav`) and get results instantly  

---

## 🧠 Machine Learning Pipeline

**Audio Input** → `.wav` files  
**Preprocessing** → Silence removal, normalization  
**Feature Extraction** → MFCC, Chroma, Spectral Centroid  
**Model Training** → SVM, RF, DT, NB using scikit-learn  
**Prediction** → Flask interface for classification  

---

## 📸 Demo Screenshots

### 🏠 Homepage UI
User lands on the homepage and uploads a machine sound `.wav` file.

![Homepage](https://github.com/Kusam-Badyal88/machine_sounds_detection/blob/master/static/screenshots/homepage.png?raw=true)

---

### ⚙️ Model Selection
User selects one of the Machine Learning models (**SVM**, **RF**, **DT**, **NB**) for prediction.

![Model Selection](https://github.com/Kusam-Badyal88/machine_sounds_detection/blob/master/static/screenshots/model_selection.png?raw=true)

---

### 🎯 Prediction Output
System predicts whether the machine sound is **Defective** or **Non-Defective** and shows model accuracy.

![Prediction](https://github.com/Kusam-Badyal88/machine_sounds_detection/blob/master/static/screenshots/prediction.png?raw=true)

---

## 🔍 How It Works

🎵 User uploads a `.wav` machine sound file  
🧪 Audio is processed and features are extracted:
- MFCC (Mel Frequency Cepstral Coefficients)  
- Chroma Frequencies  
- Spectral Centroid  

🤖 Selected ML model predicts the machine sound status  
✅ Result and model accuracy are displayed on the screen  

---

## 🧠 Model Predictions

| Label Code | Meaning           |
|------------|-------------------|
| 0          | LM_DEF 🛠️         |
| 1          | LM_NON-DEF ✅     |
| 2          | VMC_DEF ⚙️        |
| 3          | VMC_NON-DEF 🟢     |

You can group them as:

- **Defective** → Label 0 & 2  
- **Non-Defective** → Label 1 & 3

---

## 🛠️ Tech Stack

| Tool / Library     | Purpose                                  |
|--------------------|------------------------------------------|
| Python 🐍          | Programming Language                      |
| Flask 🌐           | Web Framework                             |
| Librosa 🎵         | Audio Feature Extraction                  |
| Scikit-learn 🤖     | ML Model Training & Prediction            |
| HTML/CSS 🎨        | Frontend UI Design                        |
| Pickle 🧃          | Model Saving & Loading                    |

---

## 🚀 How to Run the Project Locally

```bash
# Step 1: Clone the repository
git clone https://github.com/Kusam-Badyal88/machine_sounds_detection.git

# Step 2: Navigate to project folder
cd machine_sounds_detection

# Step 3: Install dependencies
pip install -r requirements.txt

# Step 4: Run the application
python app.py
➡️ Open your browser and visit:
http://127.0.0.1:5000

🎤 Upload your .wav machine sound file and get an instant prediction!

📂 Project Structure
cpp
Copy
Edit
machine_sounds_detection/
├── static/
│   ├── style.css
│   └── screenshots/
│       ├── homepage.png
│       ├── model_selection.png
│       └── prediction.png
├── templates/
│   ├── index.html
│   └── result.html
├── uploads/
├── svm_model.pkl
├── rf_model.pkl
├── dt_model.pkl
├── nb_model.pkl
├── app.py
├── feature_extraction.py
└── README.md
📈 Model Accuracies
Model	Accuracy
Random Forest 🌲	94%
SVM 📊	89%
Decision Tree 🌴	80%
Naive Bayes 🧠	57%

✨ Future Enhancements
🎙️ Live microphone-based detection
🤖 Add deep learning models (CNN, LSTM)
📱 Mobile app support (Android/iOS)
📊 Real-time analytics dashboard
🔊 Train on larger and more diverse machine sound datasets

## 📦 Download Large Files (Google Drive)
Due to GitHub file size limits, download the following files manually:

📥 [svm_model.pkl](https://drive.google.com/file/d/1jQ_S-p9lCunQWEohSP_mx6RsiTtZntAl/view?usp=sharing)

📥 [machine_sounds_detection.ipynb](https://drive.google.com/file/d/1wrHQWYcFGFE2NylV6zYDDtrORSTavdIY/view?usp=sharing)






