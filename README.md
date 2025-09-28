# Disease-Predictor-using-Machine-Learning

# Disease Predictor using Machine Learning

## Project Overview
The **Disease Predictor** is a machine learning-based project that predicts the **probable disease** of a patient based on **selected symptoms**. This project implements three different ML algorithms—**Decision Tree, Random Forest, and Naive Bayes**—to give accurate predictions and compare model performance.

The project also includes a **Tkinter GUI**, making it user-friendly and interactive for non-technical users.

---

## Features
- **Predict diseases** from user-selected symptoms (up to 5 at a time).  
- **Three ML algorithms implemented**:
  - Decision Tree Classifier
  - Random Forest Classifier
  - Gaussian Naive Bayes
- **Accuracy metrics** are printed in the console for evaluation.
- **GUI interface** built with Tkinter for ease of use.
- Professional and clean **layout with separate sidebar and main panel**.  

---

## Dataset
The project uses two CSV files:  

1. **Training.csv** – Used to train the machine learning models.  
2. **Testing.csv** – Used to test the accuracy of the trained models.  

**Note**: These datasets contain a list of symptoms as features and the corresponding disease as the target.

**Symptoms examples:** `back_pain`, `fever`, `headache`, `abdominal_pain`  

**Disease examples:** `Fungal infection`, `Allergy`, `Diabetes`, `Hypertension`, `Migraine`  

---

## Technologies Used
- **Python 3.x**
- **Tkinter** – For GUI development
- **Pandas & Numpy** – Data manipulation and processing
- **Scikit-learn** – Machine learning algorithms
  - DecisionTreeClassifier
  - RandomForestClassifier
  - GaussianNB
- **Matplotlib / Plotly** (Optional for visualization)
  
---

## Installation & Setup

1. **Clone the repository**:
```bash
git clone <repository_link>
cd Disease-predictor-using-ML
