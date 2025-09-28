import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib

st.set_page_config(page_title="Productivity Prediction", page_icon="🧵")

# -------- 1. Charger et préparer les données --------
@st.cache_data
def load_and_train():
    try:
        df = pd.read_csv("data.csv")
    except FileNotFoundError:
        st.error("⚠️ Fichier data.csv introuvable. Vérifie qu'il est bien dans ton repo GitHub.")
        return None, [], [], None, None

    df = df.dropna(subset=['actual_productivity']).reset_index(drop=True)
    X = df.drop(columns=['actual_productivity', 'date'])
    y = df['actual_productivity']

    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()

    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

    model = Pipeline([
        ('pre', preprocessor),
        ('model', LinearRegression())
    ])

    model.fit(X, y)
    y_pred = model.predict(X)
    return model, numeric_features, categorical_features, y, y_pred

model, num_feats, cat_feats, y_true, y_pred = load_and_train()

if model:
    # -------- 2. Interface utilisateur --------
    st.title("🚀 Prédiction de la productivité des équipes")
    st.markdown("""
    Cette application prédit la **productivité réelle** (`actual_productivity`) 
    d'une équipe d'usine de confection à partir de ses caractéristiques.
    """)

    # Inputs utilisateur
    st.subheader("➡️ Entrez les caractéristiques de l'équipe")
    inputs = {}
    for feat in num_feats:
        inputs[feat] = st.number_input(f"{feat}", value=0.0)
    for feat in cat_feats:
        inputs[feat] = st.text_input(f"{feat}", "unknown")

    if st.button("Prédire"):
        X_new = pd.DataFrame([inputs])
        y_new_pred = model.predict(X_new)[0]
        st.success(f"✅ Productivité prédite : {y_new_pred:.2f}")

    # -------- 3. Visualisation du modèle --------
    st.subheader("📊 Réel vs Prédit")
    fig, ax = plt.subplots()
    sns.scatterplot(x=y_true, y=y_pred, ax=ax)
    ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    ax.set_xlabel("Productivité réelle")
    ax.set_ylabel("Productivité prédite")
    ax.set_title("Réel vs Prédit")
    st.pyplot(fig)

    # Coefficients du modèle
    st.subheader("📈 Coefficients du modèle")
    try:
        coefs = model.named_steps['model'].coef_
        cat_features_encoded = model.named_steps['pre'].named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(cat_feats)
        feature_names = num_feats + list(ca



