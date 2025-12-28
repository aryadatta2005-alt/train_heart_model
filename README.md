# Heart Disease Prediction System 🫀

## 📌 Project Overview
This project builds a Machine Learning model to predict the likelihood of a patient having heart disease based on medical attributes (e.g., cholesterol, blood pressure, heart rate). 

Unlike many standard implementations on this dataset, this project **rigorously cleans the data** to ensure real-world applicability.

## 🚨 The "Data Leakage" Problem
During Exploratory Data Analysis (EDA), I discovered a critical issue in the raw dataset: **Duplicate Rows**.
* **Total Rows:** 986
* **Duplicates Found:** 684
* **Unique Rows:** 302

**Why this matters:**
Training on the raw data (with duplicates) artificially inflates accuracy to ~98.5% because the model sees the same patients in both the training and testing sets. This is known as **Data Leakage**.

## 🛠️ My Solution
To build a scientifically valid model, I implemented a cleaning step to remove all duplicates before training.

* **Algorithm:** Random Forest Classifier
* **Raw Accuracy (Inflated):** 98.48% (Misleading)
* **Cleaned Accuracy (Real):** **83.61%** (Validated)

This result accurately reflects how the model would perform on **new, unseen patients**.

## ⚙️ Tech Stack
* **Python**
* **Pandas** (Data Cleaning & Deduplication)
* **Scikit-Learn** (Random Forest, Grid Search)
* **Matplotlib/Seaborn** (Visualization)

## 📊 Results
On the cleaned test set, the model achieved:
* **Accuracy:** 83.61%
* **False Positives:** 7 (Safe bias: better to flag healthy people than miss sick ones)
* **False Negatives:** 3 (Low miss rate)

## 🚀 How to Run
1. Clone the repo:
   ```bash
   git clone [https://github.com/aryadatta2005-alt/train_heart_model.git](https://github.com/aryadatta2005-alt/train_heart_model.git)