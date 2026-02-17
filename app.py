'''import streamlit as st
import pandas as pd
import joblib

# App title
st.title("Mushroom Classification App")

st.write("Upload a CSV file to predict whether mushrooms are edible or poisonous.")

# Model selection dropdown
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

# Function to load selected model
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


# File uploader
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:

    # Read uploaded dataset
    data = pd.read_csv(uploaded_file)

    st.write("Uploaded Dataset:")
    st.write(data.head())

    # Load model
    model = load_model(model_name)

    # Apply SAME encoding as training
    data_encoded = pd.get_dummies(data)

    # Get training columns
    training_columns = model.feature_names_in_

    # Add missing columns
    for col in training_columns:
        if col not in data_encoded.columns:
            data_encoded[col] = 0

    # Ensure correct order
    data_encoded = data_encoded[training_columns]

    # Predict
    predictions = model.predict(data_encoded)

    # Add predictions to original dataset
    data["Prediction"] = predictions

    st.write("Prediction Results:")
    st.write(data)'''



import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# App title
st.title("Mushroom Classification App")

st.write("Upload a CSV file to predict whether mushrooms are edible or poisonous.")

# Model selection dropdown
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

# Function to load selected model
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


# File uploader
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:

    # Read uploaded dataset
    data = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Dataset Preview")
    st.write(data.head())

    # Separate target and features
    if "class" in data.columns:
        y_true = data["class"]
        X = data.drop("class", axis=1)
    else:
        y_true = None
        X = data.copy()

    # Load model
    model = load_model(model_name)

    # Apply SAME encoding as training
    X_encoded = pd.get_dummies(X)

    # Get training columns
    training_columns = model.feature_names_in_

    # Add missing columns
    for col in training_columns:
        if col not in X_encoded.columns:
            X_encoded[col] = 0

    # Ensure correct order
    X_encoded = X_encoded[training_columns]

    # Predict
    predictions = model.predict(X_encoded)

    # Add predictions to original dataset
    data["Prediction"] = predictions

    st.subheader("Prediction Results")
    st.write(data)

    # -------------------------------
    # Show metrics and confusion matrix
    # -------------------------------
    if y_true is not None:

        st.subheader("Model Performance Metrics")

        accuracy = accuracy_score(y_true, predictions)
        precision = precision_score(y_true, predictions, pos_label="p")
        recall = recall_score(y_true, predictions, pos_label="p")
        f1 = f1_score(y_true, predictions, pos_label="p")

        st.write(f"Accuracy: {accuracy:.4f}")
        st.write(f"Precision: {precision:.4f}")
        st.write(f"Recall: {recall:.4f}")
        st.write(f"F1 Score: {f1:.4f}")

        # Confusion Matrix
        st.subheader("Confusion Matrix")

        cm = confusion_matrix(y_true, predictions)

        fig, ax = plt.subplots()

        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=["Edible", "Poisonous"]
        )

        disp.plot(ax=ax)

        st.pyplot(fig)



