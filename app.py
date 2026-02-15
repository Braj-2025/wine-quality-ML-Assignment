# Let's create app.py for Streamlit
!pip install streamlit
import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

st.title("Wine Quality Classification")

uploaded_file = st.file_uploader("Upload Test Dataset (CSV)", type=["csv"])


model_name = st.selectbox("Select Model",
("Logistic Regression","Decision Tree","kNN","Naive Bayes","Random Forest","XGBoost"))

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    data = pd.read_csv(uploaded_file, sep=';')

    X = data.drop('quality', axis=1)
    y = (data['quality'] >= 7).astype(int)

    scaler = joblib.load("scaler.pkl")
    X = scaler.transform(X)

    model = joblib.load(f"{model_name}.pkl")
    preds = model.predict(X)

    st.subheader("Classification Report")
    st.text(classification_report(y, preds))

    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y, preds)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    st.pyplot(fig)