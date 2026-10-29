# 🌱 Smart Crop Recommendation System

An AI-powered Crop Recommendation web application built using **Machine Learning (Random Forest Classifier)** and **Flask**. It recommends the optimal crop to cultivate based on soil parameters and environmental weather conditions with **99.3% accuracy**.

---

## 🚀 Features
- **Accurate ML Model**: Trained with Random Forest Classifier on 2,200 agricultural data points across 22 crops.
- **Interactive Web Interface**: Clean, responsive UI with quick-fill presets (Rice, Coffee, Maize, Apple, Cotton).
- **Comprehensive Output**: Provides recommended crop, confidence percentages, crop description, and alternative crop suggestions.

---

## 📊 Parameters Used for Prediction
1. **Nitrogen (N)** - Soil Nitrogen ratio (0 - 140)
2. **Phosphorus (P)** - Soil Phosphorus ratio (5 - 145)
3. **Potassium (K)** - Soil Potassium ratio (5 - 205)
4. **Temperature** - Ambient temperature in °C (8°C - 45°C)
5. **Humidity** - Relative humidity in % (14% - 100%)
6. **Soil pH** - Soil acidity/alkalinity level (3.5 - 9.9)
7. **Rainfall** - Annual rainfall in mm (20mm - 300mm)

---

## 🛠️ Tech Stack
- **Backend**: Python 3, Flask
- **Machine Learning**: Scikit-Learn, NumPy, Pandas
- **Frontend**: HTML5, Vanilla CSS3, Google Fonts (*Plus Jakarta Sans*)

---

## 💻 How to Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/raushni-0118/Crop-Recommendation.git
cd Crop-Recommendation
```

### 2. Create and activate a Virtual Environment
```bash
# On macOS/Linux
python3 -m venv venv
source venv/bin/activate

# On Windows
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Start the Application
```bash
python app.py
```

Open your browser and navigate to:
```
http://127.0.0.1:5001
```

---

## 📁 Project Structure
```
Crop-Recommendation/
├── app.py                         # Flask web server & prediction logic
├── model.pkl                      # Trained Random Forest model
├── minmaxscaler.pkl               # Fitted MinMaxScaler
├── standscaler.pkl                # Fitted StandardScaler
├── Crop_recommendation.csv        # Dataset
├── crop_recommendation_system.ipynb # Training notebook
├── requirements.txt               # Dependencies
├── .gitignore                     # Git ignore rules
├── templates/
│   └── index.html                 # Web page template
└── static/
    └── style.css                  # UI styling & animations
```