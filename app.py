import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# =============================
# App Title
# =============================

st.title("🍄 Mushroom Classification App")

st.write(
    "Upload the Mushroom dataset CSV file to predict whether mushrooms are "
    "Edible or Poisonous using trained ML models."
)

# =============================
# Model Selection
# =============================

model_name = st.selectbox(
    "Select Model",
    [
        "Logistic Regression",
        "Decision Tree",
        "KNN",
        "Naive Bayes",
        "Random Forest",
        "XGBoost"
    ]
)

# =============================
# Load Model Function
# =============================

def load_model(name):

    if name == "Logistic Regression":
        return joblib.load("models/logistic_regression_model.pkl")

    elif name == "Decision Tree":
        return joblib.load("models/decision_tree_model.pkl")

    elif name == "KNN":
        return joblib.load("models/knn_model.pkl")

    elif name == "Naive Bayes":
        return joblib.load("models/naive_bayes_model.pkl")

    elif name == "Random Forest":
        return joblib.load("models/random_forest_model.pkl")

    elif name == "XGBoost":
        return joblib.load("models/xgboost_model.pkl")


# =============================
# File Upload
# =============================

uploaded_file = st.file_uploader("Upload Mushroom Dataset CSV", type=["csv"])

if uploaded_file is not None:

    # Read dataset
    data = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Dataset Preview")
    st.write(data.head())

    # Load model
    model = load_model(model_name)

    # =============================
    # Encode features same as training
    # =============================

    data_encoded = pd.get_dummies(data)

    # Remove class column if present
    if 'class' in data_encoded.columns:
        data_encoded = data_encoded.drop('class', axis=1)

    # Get training columns
    training_columns = model.feature_names_in_

    # Add missing columns
    for col in training_columns:
        if col not in data_encoded.columns:
            data_encoded[col] = 0

    # Ensure correct order
    data_encoded = data_encoded[training_columns]

    # =============================
    # Prediction
    # =============================

    predictions = model.predict(data_encoded)

    # Add numeric prediction
    data["Prediction"] = predictions

    # Add readable prediction
    data["Prediction Label"] = data["Prediction"].map({
        0: "Edible",
        1: "Poisonous"
    })

    st.subheader("Prediction Results")
    st.write(data)

    # =============================
    # Performance Metrics
    # =============================

    if 'class' in data.columns:

        # Convert actual labels to numeric
        y_true = data['class'].map({
            'e': 0,
            'p': 1
        })

        y_pred = predictions

        st.subheader("Model Performance Metrics")

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)

        col1, col2 = st.columns(2)

        col1.metric("Accuracy", f"{accuracy:.4f}")
        col1.metric("Precision", f"{precision:.4f}")

        col2.metric("Recall", f"{recall:.4f}")
        col2.metric("F1 Score", f"{f1:.4f}")

        # =============================
        # Confusion Matrix
        # =============================

        st.subheader("Confusion Matrix")

        cm = confusion_matrix(y_true, y_pred)

        fig, ax = plt.subplots()

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Edible", "Poisonous"],
            yticklabels=["Edible", "Poisonous"],
            ax=ax
        )

        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Confusion Matrix")

        st.pyplot(fig)

    else:

        st.warning(
            "Dataset does not contain 'class' column. "
            "Performance metrics cannot be calculated."
        )
