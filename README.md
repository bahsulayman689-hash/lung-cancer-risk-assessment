# lung-cancer-risk-assessment
An intelligent clinical patient screening dashboard that uses a trained RandomForest Classifier to compute lung cancer risk probabilities based on physiological symptoms and lifestyle factors. Built with Python and Streamlit.
# 🫁 Lung Cancer Risk Assessment App

An intelligent web application and clinical patient screening dashboard that utilizes a trained **Random Forest Classifier** pipeline to compute lung cancer clinical risk probabilities based on physiological symptoms and lifestyle factors.

Built with **Python**, **Streamlit**, and **Scikit-Learn** by **Sulayman Bah**.

---

## 🚀 Live Demo
* **App URL:** [Insert your deployed Streamlit Cloud link here]
* **Developer Portfolio:** [GitHub](https://github.com) | [LinkedIn](https://linkedin.com)

---

## 🛠️ Features & Tech Stack
- **Ensemble Random Forest Engine:** Built using `scikit-learn` to reliably process tabular clinical attributes.
- **Interactive Streamlit UI:** Designed with a clean green theme, automated dual-pane input columns, and a custom persistent sidebar layout.
- **Automated Data Processing:** Utilizes explicit column label encoding and parameter scaling tracking back to the original training metrics.
- **Dynamic Risk Reporting:** Instantly computes exact diagnostic risk percentage scores and color-coded risk categorizations.

---

## 📂 Project Structure
```text
├── lung_app.py               # Main Streamlit web application script
├── train.py                  # Model training, analysis, and pipeline export script
├── requirements.txt          # Python dependency specifications
├── README.md                 # Project documentation text file
├── lung_assets.pkl           # Exported pipeline package (Model, Scaler, Encoders)
├── infectious-diseases.png   # Optional: System Navigation Logo image
└── IMG-20260704-WA0633.jpg   # Optional: App Operator Profile avatar image
```

---

## 🔧 Installation & Local Setup

### 1. Clone the Repository
```bash
git clone https://github.com
cd lung-cancer-predictor
```

### 2. Set Up a Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Required Dependencies
```bash
pip install -r requirements.txt
```

### 4. Train the Model Pipeline
Ensure your dataset file (`survey lung cancer.csv`) is in the main directory, then run the script to output the serialized artifact:
```bash
python train.py
```

### 5. Launch the Web App
```bash
streamlit run lung_app.py
```

---

## 📊 Evaluation Schema
The model evaluates patient data points divided across two diagnostic panes:
- **Demographics & Lifestyle:** Age, Gender, Smoking History, Peer Pressure, Alcohol Consuming.
- **Clinical Indicators:** Yellow Fingers, Anxiety, Chronic Disease, Fatigue, Allergies, Wheezing, Coughing, Shortness of Breath, Swallowing Difficulty, Chest Pain.

---

## 🤝 Contact & Collaborations
I am a machine learning and deep learning engineer focused on data science, predictive analytics, and building intelligent web applications. Feel free to reach out for questions, collaborations, or feedback!

- **Name:** Sulayman Bah
- **Email:** bahsulayman689@gmail.com
- **GitHub:** [github.com]()
- **LinkedIn:** [linkedin.com](https://linkedin.com)
