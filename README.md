# 🎧 Machine Sound Detection System 🔍

An intelligent **audio classification web app** that detects defective and non-defective machine sounds using **Machine Learning**.  
This project helps identify early signs of machine faults through sound, supporting predictive maintenance in industrial setups.

---

## 🌟 Key Features

- 🛠️ **Real-time audio classification** of machine sounds  
- 🤖 **Multiple ML Models** supported: SVM, Random Forest, Decision Tree, Naive Bayes  
- 📊 **Displays Prediction and Model Accuracy**  
- 🌐 **User-Friendly Web Interface** using Flask  
- 🎵 **Advanced Feature Extraction** with MFCC, Chroma, and Spectral Centroid  
- 📁 **Upload Machine Audio** (.wav format) and get results instantly  

---

## 🔍 How It Works

1. 🎵 User uploads a `.wav` machine sound file  
2. 🧪 Features extracted: MFCC, Chroma, Spectral Centroid  
3. 🤖 Model (SVM, RF, DT, NB) predicts label  
4. ✅ Result and model accuracy shown on screen  

---

## 📂 Labels Used for Classification

| Label Code | Description         |
|------------|---------------------|
| 0          | LM_DEF 🛠️           |
| 1          | LM_NON-DEF ✅        |
| 2          | VMC_DEF ⚙️          |
| 3          | VMC_NON-DEF 🟢       |

> You can group them as:  
> - **Defective** → Label 0 & 2  
> - **Non-Defective** → Label 1 & 3  

---

## 🛠️ Tech Stack

| Tool            | Description                      |
|------------------|----------------------------------|
| Python 🐍         | Programming Language              |
| Flask 🌐          | Web Framework                    |
| Librosa 🎵        | Audio Feature Extraction         |
| Scikit-learn 🤖   | ML Model Training and Prediction |
| HTML/CSS 🎨       | UI/UX Styling                    |
| Pickle 🧃         | Save/Load Trained Models         |

---

## 🚀 Run This Project Locally

```bash
# Step 1: Clone the repository
git clone https://github.com/Kusam-Badyal88/machine_sounds_detection.git

# Step 2: Navigate to project directory
cd machine_sounds_detection

# Step 3: Install dependencies
pip install -r requirements.txt

# Step 4: Run the application
python app.py
➡️ Open your browser and go to:
http://127.0.0.1:5000

## 📸 Screenshots

### 🏠 Homepage UI  
📷 [Click to view full image](https://raw.githubusercontent.com/Kusam-Badyal88/machine_sounds_detection/master/static/screenshots/homepage.png)  
User lands on the homepage and uploads a machine sound `.wav` file.

![Homepage Screenshot](https://raw.githubusercontent.com/Kusam-Badyal88/machine_sounds_detection/master/static/screenshots/homepage.png)

---

### ⚙️ Model Selection  
📷 [Click to view full image](https://raw.githubusercontent.com/Kusam-Badyal88/machine_sounds_detection/master/static/screenshots/model_selection.png)  
User selects the ML model for prediction.

![Model Selection Screenshot](https://raw.githubusercontent.com/Kusam-Badyal88/machine_sounds_detection/master/static/screenshots/model_selection.png)

---

### 🎯 Prediction Output  
📷 [Click to view full image](https://raw.githubusercontent.com/Kusam-Badyal88/machine_sounds_detection/master/static/screenshots/prediction.png)  
Model output with prediction label and accuracy.

![Prediction Screenshot](https://raw.githubusercontent.com/Kusam-Badyal88/machine_sounds_detection/master/static/screenshots/prediction.png)




📈 Model Accuracies
Model	Accuracy
Random Forest 🌲	94%
SVM 📊	89%
Decision Tree 🌴	80%
Naive Bayes 🧠	57%

✨ Future Enhancements
🎙️ Live microphone-based detection

🤖 Deep Learning integration (CNN, LSTM)

📱 Mobile app support

📊 Real-time analytics dashboard

🔊 Larger and diverse dataset for training

📦 Download Large Files (Google Drive)
Due to GitHub file size limits, download the following files manually:

📥 svm_model.pkl

📥 machine_sounds_detection.ipynb


