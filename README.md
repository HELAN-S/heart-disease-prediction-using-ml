# heart-disease-prediction-using-ml

# ❤️ Heart Disease Prediction using Machine Learning & Ensemble Models

## 📌 Project Overview
This project focuses on predicting the presence of heart disease using multiple **machine learning and ensemble techniques**.  
The objective is to build a robust and interpretable classification system that can assist in **early diagnosis** by analyzing patient health parameters.

The project includes **end-to-end Data Science workflow**:
- Data preprocessing & cleaning
- Exploratory Data Analysis (EDA)
- Feature selection & multicollinearity analysis
- Model training, evaluation & comparison
- Ensemble learning
- Model explainability using SHAP
- Final model selection and saving

---

## 📊 Dataset
- **Dataset Name:** Cardiovascular Disease Dataset  
- **Target Variable:** `target`  
  - `1` → Heart Disease  
  - `0` → No Heart Disease  

### Data Cleaning Steps
- Removed non-informative identifier (`patientid`)
- Replaced invalid zero values using median imputation
- Dropped low-correlation features after correlation analysis
- Checked multicollinearity using **Variance Inflation Factor (VIF)**

---

## 🔍 Exploratory Data Analysis (EDA)
- Target class distribution analysis
- Feature-wise histograms grouped by target
- Boxplots for outlier detection
- Correlation heatmap
- Feature skewness analysis

---

## 🧠 Models Implemented

### 🔹 Individual Models
- Logistic Regression
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)
- Decision Tree
- Random Forest
- XGBoost
- Gradient Boosting

### 🔹 Ensemble Models
- Voting Classifier (Soft Voting)
- Bagging Classifier
- Stacking Classifier

---

## ⚙️ Feature Engineering & Importance
- Correlation-based feature selection
- Permutation Importance (Random Forest)
- SHAP (SHapley Additive exPlanations) for:
  - XGBoost
  - Random Forest
  - Logistic Regression

---

## 📈 Model Evaluation Metrics
Each model is evaluated using:
- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC
- Confusion Matrix

Comparative performance visualizations are provided using:
- Bar plots (Accuracy, ROC-AUC, F1-score)
- Confusion matrix heatmaps

---

## 🏆 Best Model Selection
Models are ranked based on **ROC-AUC**, and the best-performing model is selected.

✔ The best model is **saved using `joblib`** for future inference.

```python
joblib.dump(best_model, "best_model.joblib")

he final system supports prediction on new patient data by:

Ensuring feature alignment

Applying scaling

Generating prediction probabilities

🛠️ Tools & Technologies Used

Programming: Python

Libraries:

Pandas, NumPy

Matplotlib, Seaborn

Scikit-learn

XGBoost

SHAP

Statsmodels

Platform: Google Colab

Model Persistence: Joblib
