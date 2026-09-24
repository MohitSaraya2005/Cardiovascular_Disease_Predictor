# 🫀 CardioPulse AI — Cardiovascular Disease Risk Intelligence Platform

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io)
[![Python](https://img.shields.io/badge/Python-3.9%20%7C%203.10%20%7C%203.11-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-Decision%20Tree-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end, clinical-grade cardiovascular disease prediction and risk intelligence web platform powered by Machine Learning and built with **Streamlit**, **Scikit-Learn**, and **Plotly**.

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Dataset & Clinical Parameters](#-dataset--clinical-parameters)
- [Machine Learning Architecture](#-machine-learning-architecture)
- [Project Structure](#-project-structure)
- [Local Installation & Setup](#-local-installation--setup)
- [Deployment on Streamlit Cloud](#-deployment-on-streamlit-cloud)
- [Tech Stack](#-tech-stack)
- [Clinical Disclaimer](#-clinical-disclaimer)
- [Author & Acknowledgments](#-author--acknowledgments)

---

## 🔬 Overview

Cardiovascular diseases (CVDs) are the leading cause of death globally. Early diagnosis and proactive risk stratification can significantly improve clinical outcomes and reduce mortality. 

**CardioPulse AI** provides healthcare professionals and individuals with a modern clinical intelligence dashboard that evaluates patient biometrics, vitals, and lifestyle risk factors to estimate the likelihood of cardiovascular disease in real time.

---

## ✨ Key Features

- **🎯 Interactive Risk Assessment:**
  - Enter patient clinical indicators (Age, Gender, Height, Weight, Blood Pressure, Cholesterol, Glucose, Lifestyle habits).
  - Real-time dynamic **Body Mass Index (BMI)** calculation and classification.
  - Interactive **Plotly Gauge Chart** displaying risk probability score.
  
- **📊 Clinical Breakdown & Recommendations:**
  - Automated stratification into Low, Moderate, or High cardiovascular risk tiers.
  - Tailored clinical lifestyle and medical guidance based on patient inputs.

- **📑 Automated Clinical Report Generation:**
  - Instant summary of patient vitals and prediction outcome ready for clinical recordkeeping.

- **📈 Visual Analytics & Insights:**
  - Modern, responsive medical UI design with dark/light clinical themes and sleek typography.

- **🧭 Multi-Page Intuitive Navigation:**
  - `Home`: Platform overview, metrics summary, and cardiovascular statistics.
  - `Risk Assessment`: Real-time prediction tool with custom parameters.
  - `About`: Model hyperparameters, dataset architecture, and methodology.
  - `Contact Us`: Feedback, inquiries, and developer info.

---

## 📋 Dataset & Clinical Parameters

The predictive model is trained on a comprehensive cardiovascular dataset (`cardio_train.csv`) containing 70,000 patient records across 11 key diagnostic features:

| Feature | Type | Description | Unit / Range |
| :--- | :--- | :--- | :--- |
| **Age** | Objective | Patient age in years | 10 – 100 yrs |
| **Gender** | Objective | Biological sex | Female / Male |
| **Height** | Objective | Patient height | cm |
| **Weight** | Objective | Patient weight | kg |
| **Systolic BP (`ap_hi`)** | Examination | Systolic blood pressure | mmHg |
| **Diastolic BP (`ap_lo`)** | Examination | Diastolic blood pressure | mmHg |
| **Cholesterol** | Examination | Serum cholesterol levels | Normal / Above Normal / Well Above Normal |
| **Glucose** | Examination | Blood glucose levels | Normal / Above Normal / Well Above Normal |
| **Smoke** | Subjective | Smoking habit | Yes / No |
| **Alcohol** | Subjective | Alcohol consumption | Yes / No |
| **Active** | Subjective | Regular physical activity | Yes / No |
| **Target (`cardio`)** | Outcome | Presence or absence of CVD | 0 (No Disease) / 1 (Disease Present) |

---

## 🧠 Machine Learning Architecture

- **Algorithm:** Decision Tree Classifier (`cardiovascular_decision_tree.pkl`)
- **Preprocessing:** `StandardScaler` for numerical feature normalization (`age`, `height`, `weight`, `ap_hi`, `ap_lo`).
- **Inference Pipeline:**
  1. Input parameter validation and unit normalization.
  2. Feature vector scaling with dataset-calibrated statistics.
  3. Model evaluation generating binary classification and risk probabilities.
  4. Dynamic generation of medical risk metrics and visual gauge indicators.

---

## 📁 Project Structure

```bash
Cardiovascular_Disease_Predictor/
├── .gitignore                         # Excluded temporary & local files
├── README.md                          # Comprehensive project documentation
├── app.py                             # Main Streamlit application (Frontend & ML Pipeline)
├── cardio_train.csv                   # Cardiovascular training & calibration dataset
├── cardiovascular_decision_tree.pkl   # Serialized pre-trained Decision Tree model
├── heart_banner.jpg                   # Clinical UI header visual asset
└── requirements.txt                   # Production dependencies for deployment
```

---

## 🚀 Local Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/MohitSaraya2005/Cardiovascular_Disease_Predictor.git
cd Cardiovascular_Disease_Predictor
```

### 2. Create and Activate a Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit Application
```bash
streamlit run app.py
```

The app will launch in your default web browser at `http://localhost:8501`.

---

## ☁️ Deployment on Streamlit Cloud

To deploy this project to **Streamlit Community Cloud**:

1. **Push your code to GitHub** (Ensure `app.py`, `cardiovascular_decision_tree.pkl`, `requirements.txt`, and `cardio_train.csv` are committed).
2. Go to [share.streamlit.io](https://share.streamlit.io/) and log in with your GitHub account.
3. Click **"New app"**.
4. Select your repository: `MohitSaraya2005/Cardiovascular_Disease_Predictor`.
5. Set the Main file path to: `app.py`.
6. Click **"Deploy!"** 🚀

---

## 🛠️ Tech Stack

- **Frontend & App Framework:** [Streamlit](https://streamlit.io/)
- **Machine Learning:** [Scikit-Learn](https://scikit-learn.org/), [Joblib](https://joblib.readthedocs.io/)
- **Data Manipulation:** [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/)
- **Data Visualization:** [Plotly Express & Graph Objects](https://plotly.com/python/)
- **Design & Layout:** Custom CSS, Glassmorphism, Google Fonts (`Plus Jakarta Sans`)

---

## ⚠️ Clinical Disclaimer

> **IMPORTANT:** CardioPulse AI is intended solely for educational, research, and preliminary screening assistance. It does **NOT** constitute medical advice, formal clinical diagnosis, or replacement for consultation with a qualified medical professional. Patients experiencing acute symptoms should seek emergency medical attention immediately.

---

## 👨‍💻 Author

**Mohit Saraya**  
- GitHub: [@MohitSaraya2005](https://github.com/MohitSaraya2005)
- Repository: [Cardiovascular_Disease_Predictor](https://github.com/MohitSaraya2005/Cardiovascular_Disease_Predictor)

---
*⭐ If you find this project useful, feel free to give it a star on GitHub!*
