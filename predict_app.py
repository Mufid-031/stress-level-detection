import streamlit as st
import pandas as pd
import joblib

st.subheader("🔮 Prediksi Data Baru")
model = joblib.load("model.pkl")

x = st.number_input("x")
y_val = st.number_input("y")
z = st.number_input("z")
bvp = st.number_input("bvp")
eda = st.number_input("eda")
hr = st.number_input("hr")

if st.button("Prediksi Stress"):
    input_data = pd.DataFrame([[x, y_val, z, bvp, eda, hr]], columns=['x','y','z','bvp','eda','hr'])
    pred = model.predict(input_data)[0]
    st.success(f"Prediksi: LABEL = {pred}")