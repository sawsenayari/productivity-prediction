import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

st.set_page_config(page_title="Productivity Prediction", page_icon="🧵")

# -------- 1. Charger et préparer les données --------
@st.cache_data
def load_and_train():
    try:
        df = pd.read_csv("data.csv")
    except FileNotFoundError:
        st.error("⚠️ Fichier data.csv introuvable. Vérifie qu'il est bien dans ton repo GitHub.")
        return None, [], []

    # Supprimer les lignes sans target
    df = df.dropna(subset=['actual_productivity']).reset_index(drop=True)

    # Features & target
    X = df.drop(columns=['actual_productivity', 'date'])
    y = df['actual_productivity']

    # Séparer numériques et catégorielles
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()

    # Transformers
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # Preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # Pipeline complet
    model = Pipeline(steps=[('pre', preprocessor),
                            ('model', LinearRegression())])

    model.fit(X, y)
    return model, numeric_features, categorical_features

# Charger modèle
model, num_feats, cat_feats = load_and_train()

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

    # -------- 3. Prédiction --------
    if st.button("Prédire"):
        X_new = pd.DataFrame([inputs])
        y_pred = model.predict(X_new)[0]
        st.success(f"✅ Productivité prédite : {y_pred:.2f}")


