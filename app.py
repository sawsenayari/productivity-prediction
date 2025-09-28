import streamlit as st
import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import os

st.set_page_config(page_title="Productivity Prediction", page_icon="🧵")

# -------- 1. Charger ou entraîner le modèle --------
MODEL_FILE = "model.pkl"

@st.cache_data
def load_model():
    if os.path.exists(MODEL_FILE):
        return joblib.load(MODEL_FILE)
    else:
        try:
            df = pd.read_csv("data.csv")
        except FileNotFoundError:
            st.error("⚠️ Fichier data.csv introuvable.")
            return None
        
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
        joblib.dump(model, MODEL_FILE)  # sauvegarder le modèle
        return model

model = load_model()

if model:
    st.title("🚀 Prédiction de la productivité des équipes")
    st.markdown("""
        Cette application prédit la **productivité réelle** (`actual_productivity`) 
        d'une équipe à partir de ses caractéristiques.
    """)

    # Récupérer les features
    df = pd.read_csv("data.csv")
    X = df.drop(columns=['actual_productivity', 'date'])
    num_feats = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_feats = X.select_dtypes(include=['object']).columns.tolist()

    st.subheader("➡️ Entrez les caractéristiques de l'équipe")

    inputs = {}
    for feat in num_feats:
        inputs[feat] = st.number_input(f"{feat}", value=0.0)
    for feat in cat_feats:
        options = df[feat].dropna().unique().tolist()
        inputs[feat] = st.selectbox(f"{feat}", options)

    if st.button("Prédire"):
        X_new = pd.DataFrame([inputs])
        y_pred = model.predict(X_new)[0]
        st.success(f"✅ Productivité prédite : {y_pred:.2f}")

