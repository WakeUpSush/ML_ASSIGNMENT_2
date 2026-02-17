# Mushroom Classification using Machine Learning and Streamlit

## 1. Problem Statement

The goal of this project is to build multiple machine learning classification models to predict whether a mushroom is **edible** or **poisonous** based on its physical characteristics.

This project demonstrates an end-to-end machine learning workflow including:

- Dataset selection and preprocessing
- Training multiple classification models
- Model evaluation using multiple performance metrics
- Model comparison
- Building an interactive Streamlit web application
- Deployment-ready ML system

---

## 2. Dataset Description

Dataset Used: **UCI Mushroom Dataset**

Source: UCI Machine Learning Repository

Dataset characteristics:

- Total instances: 8124
- Total features: 22 categorical features
- Target variable: `class`
  - e → edible
  - p → poisonous

Feature examples:

- cap-shape
- cap-color
- odor
- gill-size
- stalk-shape
- habitat
- population
- etc.

All features are categorical and were converted into numerical format using one-hot encoding.

This dataset is well-suited for classification tasks and satisfies assignment requirements:
- More than 500 instances
- More than 12 features

---

## 3. Machine Learning Models Used

The following 6 classification models were implemented:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (KNN)
4. Naive Bayes Classifier
5. Random Forest Classifier (Ensemble Model)
6. XGBoost Classifier (Ensemble Model)

---

## 4. Evaluation Metrics

Each model was evaluated using the following metrics:

- Accuracy
- AUC Score
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)

---

## 5. Model Performance Comparison

| Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|------|----------|-----|-----------|--------|----------|-----|
| Logistic Regression | 0.99 | 0.99 | 1.00 | 0.99 | 0.99 | 0.99 |
| Decision Tree | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| KNN | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| Naive Bayes | 0.99 | 0.99 | 0.99 | 0.99 | 0.99 | 0.99 |
| Random Forest | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| XGBoost | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |

(Note: Values may vary slightly depending on train-test split)

---

## 6. Model Observations

### Logistic Regression
- Performs very well on this dataset
- Works efficiently for linearly separable data
- Slightly less powerful than ensemble methods

### Decision Tree
- Achieves perfect accuracy
- Easy to interpret
- May overfit on some datasets, but works well here

### KNN
- Achieves perfect accuracy
- Works well because dataset has clear separable patterns
- Slower prediction time compared to other models

### Naive Bayes
- Very fast and efficient
- Assumes feature independence
- Slightly lower performance compared to ensemble methods

### Random Forest (Ensemble)
- Excellent performance
- Reduces overfitting
- Very robust model
- One of the best performing models

### XGBoost (Ensemble)
- Best overall performance
- Highly optimized ensemble model
- Handles complex patterns effectively

---



