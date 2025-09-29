import streamlit as st
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title="Prédiction Productivité", page_icon="🧵", layout="wide")
st.title("🚀 Prédiction de la productivité des équipes")
st.markdown("Interface simple pour prédire la productivité réelle d'une équipe avec **Random Forest**.")

# Charger et entraîner le modèle
@st.cache_data
def load_and_train():
    try:
        df = pd.read_csv("data.csv")
    except FileNotFoundError:
        st.error("⚠️ Fichier data.csv introuvable.")
        return None, [], [], None

    df = df.dropna(subset=['actual_productivity']).reset_index(drop=True)
    X = df.drop(columns=['actual_productivity', 'date'])
    y = df['actual_productivity']

    numeric_features = X.select_dtypes(include=['int64','float64']).columns.tolist()
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

    # 🔥 Ici on remplace LinearRegression par RandomForestRegressor
    model = Pipeline([
        ('pre', preprocessor),
        ('model', RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            n_jobs=-1
        ))
    ])

    model.fit(X, y)
    return model, numeric_features, categorical_features

model, num_feats, cat_feats = load_and_train()

# Sidebar pour saisir les features
st.sidebar.header("📝 Entrez les caractéristiques de l'équipe")
inputs = {}
if model:
    # Features numériques
    for feat in num_feats:
        inputs[feat] = st.sidebar.number_input(f"{feat}", value=0.0)

    # Features catégorielles
    df_ref = pd.read_csv("data.csv")
    for feat in cat_feats:
        unique_vals = ["unknown"] + sorted(df_ref[feat].dropna().unique().tolist())
        inputs[feat] = st.sidebar.selectbox(f"{feat}", unique_vals)

    # Bouton prédiction
    if st.sidebar.button("Prédire"):
        X_new = pd.DataFrame([inputs])
        y_new_pred = model.predict(X_new)[0]
        st.success(f"✅ Productivité prédite : {y_new_pred:.2f}")

