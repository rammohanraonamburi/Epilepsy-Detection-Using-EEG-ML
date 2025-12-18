# 🧠 Epilepsy Detection from EEG Signals using Machine Learning

> **A complete end-to-end machine learning pipeline for epilepsy detection using EEG & ECG signals — from raw EDF files to optimized models with genetic algorithm–based feature selection.**

---

## 🚀 Project Highlights
- Raw **EDF → CSV** EEG signal processing
- Statistical **feature extraction** from EEG & ECG channels
- Handling **class imbalance** using **SMOTE-Tomek**
- Robust normalization using **RobustScaler**
- Multiple ensemble models compared:
  - Random Forest
  - Gradient Boosting
  - XGBoost
- **Genetic Algorithm (GA)** for feature selection
- Strong, realistic performance on unseen test data

---

## 📌 Problem Statement
Epilepsy is a neurological disorder characterized by abnormal brain activity leading to seizures.  
Manual EEG interpretation is time-consuming and requires expert neurologists.

This project aims to **automate epilepsy detection** by extracting meaningful statistical features from EEG signals and training machine learning models to classify epileptic vs non-epileptic cases.

---

## 🛠️ Tech Stack
- **Language:** Python  
- **Environment:** Jupyter Notebook  
- **Libraries:**
  - NumPy, Pandas
  - SciPy
  - Scikit-learn
  - XGBoost
  - imbalanced-learn (SMOTE-Tomek)
  - geneticalgorithm
  - pyEDFlib

---

## 📂 Project Structure
```
📦 Epilepsy-Detection-Using-EEG-ML
 ┣ 📁 EEGData/
 ┣ 📁 Datasets_CSV/
 ┣ 📁 ExtractedFeatures/
 ┣ 📁 Dataset/
 ┃ ┣ train.csv
 ┃ ┣ val.csv
 ┃ ┣ test.csv
 ┃ ┣ train_balanced_scaled.csv
 ┃ ┣ val_scaled.csv
 ┃ ┗ test_scaled.csv
 ┣ 📄 features_master_dataset.csv
 ┣ 📓 Epilepsy.ipynb
 ┗ 📄 README.md
```

---

## 🔄 End-to-End Workflow

### 1️⃣ EDF → CSV Conversion
- EEG signals read using **pyEDFlib**
- All EEG & ECG channels extracted
- Time column generated using sampling frequency
- Subject-wise labels added (0 = Non-Epileptic, 1 = Epileptic)

### 2️⃣ Feature Extraction
- Signals divided into **chunks of 1000 samples**
- Extracted features per channel:
  - Mean
  - Standard Deviation
  - Minimum
  - Maximum
  - Skewness
  - Kurtosis
- **120 features per sample**

### 3️⃣ Dataset Preparation
- Combined all feature files
- Shuffled dataset
- Split:
  - 70% Training
  - 10% Validation
  - 20% Testing

### 4️⃣ Class Imbalance Handling
- Applied **SMOTE-Tomek**
- Balanced training dataset

### 5️⃣ Feature Scaling
- **RobustScaler** for skewed distributions

---

## 🤖 Model Performance (All Features)

| Model | Test Accuracy | AUC-ROC |
|------|---------------|--------|
| Random Forest | 92.40% | 0.973 |
| Gradient Boosting | 91.94% | 0.974 |
| **XGBoost (Best)** | **96.43%** | **0.992** |

---

## 🧬 Genetic Algorithm Feature Selection
- Reduced features from **120 → 50**
- Fitness: Logistic Regression validation accuracy

### Performance After Feature Selection

| Model | Test Accuracy | AUC-ROC |
|------|---------------|--------|
| Random Forest | 86.82% | 0.931 |
| Gradient Boosting | 84.65% | 0.927 |
| XGBoost | 88.06% | 0.948 |

---

## 🏆 Final Conclusion
- **XGBoost with full feature set** achieved best performance
- ~**96.4% test accuracy**
- GA improves interpretability but slightly reduces accuracy

---

## ▶️ How to Run
```bash
pip install numpy pandas scipy scikit-learn xgboost imbalanced-learn geneticalgorithm pyEDFlib
jupyter notebook Epilepsy.ipynb
```

---

## 🎯 Applications
- AI-assisted epilepsy diagnosis
- EEG signal classification
- Healthcare machine learning research


