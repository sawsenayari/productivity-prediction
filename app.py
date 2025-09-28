import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib

st.set_page_config(page_title="Productivity Prediction", page_icon="🧵", layout="wide")

# Charger et entraîner le modèle
@st.cache_data
def load_and_train():
    df = pd.read_csv("data.csv")
    df = df.dropna(subset=['actual_productivity']).reset_index(drop=True)
    X = df.drop(columns=['actual_productivity', 'date'])
    y = df['actual_productivity']

    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()

    numeric_transformer = Pipeline([('imputer', SimpleImputer(strategy='median')),
                                    ('scaler', StandardScaler())])
    categorical_transformer = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')),
                                        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])

    preprocessor = ColumnTransformer([('num', numeric_transformer, numeric_features),
                                      ('cat', categorical_transformer, categorical_features)])

    model = Pipeline([('pre', preprocessor),
                      ('model', LinearRegression())])

    model.fit(X, y)
    y_pred = model.predict(X)
    return model, numeric_features, categorical_features, y, y_pred

model, num_feats, cat_feats, y_true, y_pred = load_and_train()

# Interface
st.title("🚀 Prédiction de la productivité des équipes")
st.sidebar.header("Saisir les caractéristiques de l'équipe")

inputs = {}
for feat in num_feats:
    inputs[feat] = st.sidebar.number_input(f"{feat}", value=0.0)
for feat in cat_feats:
    inputs[feat] = st.sidebar.text_input(f"{feat}", "unknown")

if st.sidebar.button("Prédire"):
    X_new = pd.DataFrame([inputs])
    y_new_pred = model.predict(X_new)[0]
    st.sidebar.success(f"✅ Productivité prédite : {y_new_pred:.2f}")

# Courbe Réel vs Prédit
st.subheader("📊 Réel vs Prédit")
df_plot = pd.DataFrame({"Réel": y_true, "Prédit": y_pred})
fig = px.scatter(df_plot, x="Réel", y="Prédit", trendline="ols",
                 labels={"Réel":"Productivité réelle","Prédit":"Productivité prédite"},
                 title="Réel vs Prédit")
st.plotly_chart(fig, use_container_width=True)

# Coefficients interactifs
st.subheader("📈 Coefficients du modèle")
try:
    coefs = model.named_steps['model'].coef_
    cat_features_encoded = model.named_steps['pre'].named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(cat_feats)
    feature_names = num_feats + list(cat_features_encoded)
    coef_df = pd.DataFrame({"Feature": feature_names, "Coefficient": coefs})

    threshold = st.slider("Afficher les coefficients supérieurs à (valeur absolue)", 0.0, float(max(abs(coefs))), 0.1, 0.1)
    coef_df_filtered = coef_df[coef_df['Coefficient'].abs() >= threshold]
    st.dataframe(coef_df_filtered.sort_values(by='Coefficient', key=abs, ascending=False))

    fig_coef = px.bar(coef_df_filtered, x='Feature', y='Coefficient', title="Importance des coefficients", text='Coefficient')
    st.plotly_chart(fig_coef, use_container_width=True)
except:
    st.info("⚠️ Impossible d'extraire tous les coefficients pour les features catégorielles encodées.")
