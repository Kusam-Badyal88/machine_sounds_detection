# 🎧 Machine Sound Detection System 🔍

An intelligent **audio classification web app** that detects defective and non-defective machine sounds using **Machine Learning**.  
This project helps identify early signs of machine faults through sound, supporting predictive maintenance in industrial setups.

---

## 🌟 **Key Features**

- 🛠️ **Real-time audio classification** of machine sounds  
- 🤖 **Multiple ML Models** supported: SVM, Random Forest, Decision Tree, Naive Bayes  
- 📊 **Displays Prediction and Model Accuracy**  
- 🌐 **User-Friendly Web Interface** using Flask  
- 🎵 **Advanced Feature Extraction** with MFCC, Chroma, and Spectral Centroid  
- 📁 **Upload Machine Audio** (.wav format) and get results instantly  

---

## 🔍 **How It Works**

1. User uploads a `.wav` machine audio file  
2. Features are extracted from the audio:
   - **MFCC**
   - **Chroma Frequencies**
   - **Spectral Centroid**
3. Selected ML model makes a prediction  
4. Result and model accuracy are displayed on the webpage  

---

## 📂 **Labels Used for Classification**

| Label Code | Description         |
|------------|---------------------|
| 0          | LM_DEF 🛠️           |
| 1          | LM_NON-DEF ✅        |
| 2          | VMC_DEF ⚙️          |
| 3          | VMC_NON-DEF 🟢       |

> You can map them to:
> - **Defective** → Label 0 & 2  
> - **Non-Defective** → Label 1 & 3

---

## 🛠️ **Tech Stack**

| Tool           | Usage                            |
|----------------|----------------------------------|
| Python 🐍       | Core programming language         |
| Flask 🌐        | Web framework                    |
| Librosa 🎵      | Audio signal processing           |
| Scikit-learn 🤖 | ML algorithms and models          |
| HTML/CSS 🖥️     | Web UI and styling                |
| Pickle 🧃       | Saving/loading trained models      |

---

## 🚀 **Run This Project Locally**

```bash
git clone https://github.com/Kusam-Badyal88/machine_sounds_detection.git
cd machine_sounds_detection
pip install -r requirements.txt
python app.py
➡️ Open your browser and visit:
http://127.0.0.1:5000

## 🖼️ **Demo Screenshots**

| Homepage UI | Model Selection Page | Prediction Output |
|-------------|----------------------|-------------------|
| ![Homepage](https://github.com/Kusam-Badyal88/machine_sounds_detection/blob/master/static/screenshots/homepage.png) | ![Model Selection](https://github.com/Kusam-Badyal88/machine_sounds_detection/blob/master/model_selection/select_model.png?raw=true) | ![Prediction](https://github.com/Kusam-Badyal88/machine_sounds_detection/blob/master/prediction/output_result.png?raw=true) |



📈 Model  Accuracy
Model	Accuracy
Random Forest 🌲 94%
SVM 📊	 89%
Decision Tree 🌴	80%
Naive Bayes 🧠	 57%

✨ Future Enhancements
🎙️ Live microphone-based detection

🤖 Deep Learning support (e.g., CNN, LSTM)

📱 Mobile app integration

📊 Real-time dashboard for industrial monitoring

🧠 Expand dataset for higher accuracy

📦 Download Large Files (Google Drive Links)
Due to GitHub’s file size limits, model files are available for manual download:

📥 Download svm_model.pkl

📥 Download machine_sounds_detection.ipynb
