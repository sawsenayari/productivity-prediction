import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Prédiction Productivité", page_icon="🧵")

st.title("🚀 Prédiction de la productivité des équipes")
st.markdown("Interface simple pour prédire la productivité réelle d'une équipe.")

@st.cache_data
def load_and_train():
    try:
        df = pd.read_csv("data.csv")
    except FileNotFoundError:
        st.error("⚠️ Fichier data.csv introuvable.")
        return None, [], [], None, None

    df = df.dropna(subset=['actual_productivity']).reset_index(drop=True)
    X = df.drop(columns=['actual_productivity', 'date'])
    y = df['actual_productivity']

    numeric_features = X.select_dtypes(include=['int64','float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()

    numeric_transformer = Pipeline([('imputer', SimpleImputer(strategy='median')),
                                    ('scaler', StandardScaler())])
    categorical_transformer = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')),
                                        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
    
    preprocessor = ColumnTransformer([('num', numeric_transformer, numeric_features),
                                      ('cat', categorical_transformer, categorical_features)])
    
    model = Pipeline([('pre', preprocessor), ('model', LinearRegression())])
    model.fit(X, y)
    y_pred = model.predict(X)
    return model, numeric_features, categorical_features, y, y_pred

model, num_feats, cat_feats, y_true, y_pred = load_and_train()

# Sidebar pour les entrées utilisateur
st.sidebar.header("Entrer les caractéristiques de l'équipe")
inputs = {}
if model:
    for feat in num_feats:
        inputs[feat] = st.sidebar.number_input(f"{feat}", value=0.0)
    for feat in cat_feats:
        inputs[feat] = st.sidebar.text_input(f"{feat}", "unknown")
    
    if st.sidebar.button("Prédire"):
        X_new = pd.DataFrame([inputs])
        y_new_pred = model.predict(X_new)[0]
        st.sidebar.success(f"✅ Productivité prédite : {y_new_pred:.2f}")

# Graphique Réel vs Prédit
if model:
    st.subheader("📊 Réel vs Prédit")
    fig, ax = plt.subplots()
    ax.scatter(y_true, y_pred)
    ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    ax.set_xlabel("Productivité réelle")
    ax.set_ylabel("Productivité prédite")
    st.pyplot(fig)
