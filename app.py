import streamlit as st
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title="Prédiction Productivité", page_icon="🧵", layout="wide")
st.title("🚀 Prédiction de la productivité (Random Forest)")
