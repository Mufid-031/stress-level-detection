import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

# ==========================
# 🎯 CONFIGURASI DASAR
# ==========================
st.set_page_config(page_title="SMOTE Visualizer", layout="wide")

st.title("🧠 Visualisasi Data Stress Sebelum & Sesudah SMOTE")
st.write("Menampilkan distribusi data, deteksi outlier, dan hasil klasifikasi menggunakan Random Forest.")

# ==========================
# 📥 LOAD DATA
# ==========================
uploaded_file = st.file_uploader("Upload file CSV (contoh: upsample.csv)", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("✅ Data berhasil dimuat!")
    st.write(df.head())

    st.write(f"Jumlah data awal: **{len(df)}**")

    # ==========================
    # ⚙️ OUTLIER DETECTION (Z-Score)
    # ==========================
    st.subheader("⚠️ Deteksi Outlier")

    numeric_cols = ['x', 'y', 'z', 'bvp', 'eda', 'hr']
    z_scores = np.abs((df[numeric_cols] - df[numeric_cols].mean()) / df[numeric_cols].std())
    threshold = 3
    outliers = (z_scores > threshold).any(axis=1)
    outlier_count = outliers.sum()

    st.write(f"Jumlah outlier terdeteksi: **{outlier_count}**")
    st.write(f"Persentase outlier: **{(outlier_count / len(df)) * 100:.2f}%**")

    # Hapus outlier
    df_clean = df[~outliers]
    st.write(f"Jumlah data setelah hapus outlier: **{len(df_clean)}**")

    # ==========================
    # ⚙️ PREPROCESSING
    # ==========================
    X = df_clean[['x', 'y', 'z', 'bvp', 'eda', 'hr']]
    y = df_clean['LABEL']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ==========================
    # 📊 DISTRIBUSI LABEL
    # ==========================
    st.subheader("📊 Distribusi Label Sebelum & Sesudah SMOTE")

    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.countplot(x=y_train, ax=ax, palette="Set2")
        ax.set_title("Distribusi Label Sebelum SMOTE")
        st.pyplot(fig)

    # ==========================
    # 🔁 SMOTE
    # ==========================
    smote = SMOTE(random_state=42, k_neighbors=3)
    X_res, y_res = smote.fit_resample(X_train, y_train)

    with col2:
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.countplot(x=y_res, ax=ax, palette="Set1")
        ax.set_title("Distribusi Label Sesudah SMOTE")
        st.pyplot(fig)

    # ==========================
    # 🧬 PCA UNTUK VISUALISASI
    # ==========================
    pca = PCA(n_components=2, random_state=42)
    X_pca_before = pca.fit_transform(X_train)
    X_pca_after = pca.fit_transform(X_res)

    df_before = pd.DataFrame(X_pca_before, columns=["PCA1", "PCA2"])
    df_before["LABEL"] = y_train.values

    df_after = pd.DataFrame(X_pca_after, columns=["PCA1", "PCA2"])
    df_after["LABEL"] = y_res.values

    st.subheader("🔍 Visualisasi PCA Sebelum dan Sesudah SMOTE")
    col3, col4 = st.columns(2)

    with col3:
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.scatterplot(
            data=df_before.sample(min(3000, len(df_before)), random_state=42),
            x="PCA1", y="PCA2", hue="LABEL", palette="Set2", alpha=0.7
        )
        ax.set_title("Sebelum SMOTE")
        st.pyplot(fig)

    with col4:
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.scatterplot(
            data=df_after.sample(min(3000, len(df_after)), random_state=42),
            x="PCA1", y="PCA2", hue="LABEL", palette="Set1", alpha=0.7
        )
        ax.set_title("Sesudah SMOTE")
        st.pyplot(fig)

    # ==========================
    # 🤖 MODELING
    # ==========================
    st.subheader("⚙️ Training Random Forest")

    model_before = RandomForestClassifier(n_estimators=100, random_state=42)
    model_before.fit(X_train, y_train)
    y_pred_before = model_before.predict(X_test)

    model_after = RandomForestClassifier(n_estimators=100, random_state=42)
    model_after.fit(X_res, y_res)
    y_pred_after = model_after.predict(X_test)

    # ==========================
    # 📈 HASIL EVALUASI
    # ==========================
    acc_before = accuracy_score(y_test, y_pred_before)
    acc_after = accuracy_score(y_test, y_pred_after)
    f1_before = f1_score(y_test, y_pred_before, average='macro')
    f1_after = f1_score(y_test, y_pred_after, average='macro')

    st.subheader("📋 Hasil Evaluasi Model")
    col5, col6 = st.columns(2)
    with col5:
        st.metric("Akurasi (Tanpa SMOTE)", f"{acc_before:.4f}")
        st.metric("F1 Macro (Tanpa SMOTE)", f"{f1_before:.4f}")
    with col6:
        st.metric("Akurasi (Dengan SMOTE)", f"{acc_after:.4f}")
        st.metric("F1 Macro (Dengan SMOTE)", f"{f1_after:.4f}")

    with st.expander("Lihat Classification Report Lengkap"):
        st.text("📄 [Tanpa SMOTE]")
        st.text(classification_report(y_test, y_pred_before))
        st.text("📄 [Dengan SMOTE]")
        st.text(classification_report(y_test, y_pred_after))
else:
    st.info("📂 Silakan upload file CSV terlebih dahulu untuk melanjutkan.")
