# 📊 AQI Prediction with Ensemble Learning

> A machine learning project to predict Air Quality Index (AQI) using various ensemble techniques.

---

## 🚀 Overview

This project demonstrates how to build and evaluate machine learning models for **Air Quality Index (AQI) prediction**. It utilizes a synthetic dataset, simulates real-world air pollutant measurements, and employs several ensemble learning methods (Random Forest, XGBoost, Bagging, AdaBoost, and a Voting Classifier) to achieve robust predictions. The project includes data preprocessing, model training, hyperparameter tuning, and comprehensive evaluation with visualizations.

---

## ✨ Features

* **Synthetic Data Generation**: Creates a realistic-looking dataset of air pollutant measurements and calculated AQI levels.
* **Data Preprocessing**: Handles feature scaling and target encoding for model readiness.
* **Ensemble Learning**: Implements and compares:
    * **Random Forest Classifier**
    * **XGBoost Classifier**
    * **Bagging Classifier**
    * **AdaBoost Classifier**
    * **Voting Classifier** (combining the best performing models)
* **Hyperparameter Tuning**: Uses `GridSearchCV` to optimize the performance of individual models.
* **Comprehensive Evaluation**: Provides metrics and visualizations, including:
    * Accuracy Score
    * ROC AUC Score
    * Confusion Matrix
    * ROC Curves (One-vs-Rest)
    * Feature Importance Plots
    * Distribution analysis of CO levels and other features.

---

## 💻 How to Run

To run this project locally, follow these steps:

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd <your-project-directory>
```
*(Replace `<your-repository-url>` with the actual URL of your Git repository and `<your-project-directory>` with your desired project folder name.)*

---

### 2. Install Dependencies

Ensure you have Python installed. Then, install the required libraries using pip:

```bash
pip install numpy pandas scikit-learn xgboost matplotlib seaborn
```

---

### 3. Execute the Script

Run the main Python script from your terminal:

```bash
python your_script_name.py
```
*(Replace `your_script_name.py` with the actual name of your Python file, e.g., `aqi_predictor.py`)*

This will:
1.  Generate a synthetic dataset named `synthetic_aqi_data.csv`.
2.  Train and evaluate the machine learning models.
3.  Display various plots and print evaluation metrics in your console.

---

## 📊 Results and Visualizations

The script will output accuracy scores and generate several plots to help you understand the model's performance and data characteristics:

* **Model Accuracy Comparison**: Bar chart comparing the accuracies of Random Forest and XGBoost.
* **Confusion Matrix**: Visualizes the performance of the ensemble model by showing correct vs. incorrect predictions for each AQI category.
* **ROC Curve (One-vs-Rest)**: Illustrates the diagnostic ability of the binary classifiers as their discrimination threshold is varied.
* **Feature Importance**: Identifies the most influential features in predicting AQI, derived from the best Random Forest model.
* **CO Quality Distribution**: Shows the distribution of predicted Carbon Monoxide (CO) levels and their corresponding quality categories.
* **Concentrations of all features in Test Set**: Histograms for each scaled feature in the test set, showing their distributions.

---

## 🛠 Tech Stack

* **Python**: The core programming language.
* **NumPy**: For numerical operations.
* **Pandas**: For data manipulation and analysis.
* **Scikit-learn**: For machine learning models, preprocessing, and evaluation.
* **XGBoost**: For gradient boosting.
* **Matplotlib**: For creating static, interactive, and animated visualizations.
* **Seaborn**: For statistical data visualization.

---

## 🤝 Contributing

Contributions are welcome! If you have suggestions for improvements, new features, or bug fixes, please feel free to:

1.  Fork the repository.
2.  Create a new branch (`git checkout -b feature/your-feature-name`).
3.  Make your changes and commit them (`git commit -m 'Add new feature'`).
4.  Push to the branch (`git push origin feature/your-feature-name`).
5.  Open a Pull Request.

---
