import streamlit as st
import pandas as pd
import joblib

# Charger modèle
model = joblib.load("linear_regression_productivity.joblib")

st.title("🚀 Prédiction de la productivité des équipes")

targeted = st.number_input("Targeted Productivity", 0.0, 1.0, 0.8, step=0.01)
smv = st.number_input("SMV (Standard Minute Value)", 0.0, 100.0, 25.0, step=0.5)
wip = st.number_input("WIP (Work in progress)", 0, 1000, 150, step=10)
over_time = st.number_input("Overtime (minutes)", 0, 1000, 300, step=10)
incentive = st.number_input("Incentive (BDT)", 0, 10000, 500, step=50)
idle_time = st.number_input("Idle time", 0.0, 10.0, 0.0, step=0.1)
idle_men = st.number_input("Idle men", 0, 100, 0, step=1)
style_change = st.number_input("Number of style changes", 0, 10, 1, step=1)
workers = st.number_input("Number of workers", 1, 200, 50, step=1)

department = st.selectbox("Department", ["sewing","finishing","cutting"])
team = st.text_input("Team number", "5")
day_final = st.selectbox("Day", ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"])
quarter_final = st.selectbox("Quarter", ["Quarter1","Quarter2","Quarter3","Quarter4"])

# Features dérivées
wip_per_worker = wip / workers
overtime_per_worker = over_time / workers
incentive_per_worker = incentive / workers
prod_gap = 0.0

if st.button("Prédire la productivité"):
    X_new = pd.DataFrame([{
        'targeted_productivity': targeted,
        'smv': smv,
        'wip': wip,
        'over_time': over_time,
        'incentive': incentive,
        'idle_time': idle_time,
        'idle_men': idle_men,
        'no_of_style_change': style_change,
        'no_of_workers': workers,
        'department': department,
        'team': team,
        'day_final': day_final,
        'quarter_final': quarter_final,
        'wip_per_worker': wip_per_worker,
        'overtime_per_worker': overtime_per_worker,
        'incentive_per_worker': incentive_per_worker,
        'prod_gap': prod_gap
    }])
    y_pred = model.predict(X_new)[0]
    st.success(f"✅ Productivité prédite : {y_pred:.2f}")
