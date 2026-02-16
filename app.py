import streamlit as st
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
    st.write(data)


