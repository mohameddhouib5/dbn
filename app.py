# app.py
import streamlit as st
import joblib
import numpy as np
from PIL import Image
import os

# === CONFIGURATION ===
st.set_page_config(page_title="DBN MNIST", layout="centered")
st.title("Reconnaissance de Chiffres Manuscrits")
st.write("Upload une image (n'importe quelle taille) → prédiction instantanée !")

st.info("L'image sera redimensionnée à 28×28 et adaptée au format MNIST (chiffre blanc sur fond noir).")

# === CHARGER LES MODÈLES ===
@st.cache_resource
def load_models():
    try:
        rbm1 = joblib.load('models/rbm1.pkl')
        rbm2 = joblib.load('models/rbm2.pkl')
        clf = joblib.load('models/classifier.pkl')
        st.success("Modèles chargés avec succès !")
        return rbm1, rbm2, clf
    except FileNotFoundError as e:
        st.error(f"Modèle manquant : {e}")
        st.stop()
    except Exception as e:
        st.error(f"Erreur de chargement : {e}")
        st.stop()

rbm1, rbm2, clf = load_models()

# === PRÉ-TRAITEMENT INTELLIGENT ===
def preprocess_and_predict(image):
    # 1. Convertir en gris + redimensionner
    img = image.convert('L')
    img = img.resize((28, 28))
    img_array = np.array(img, dtype=np.float32)

    # 2. INVERSION AUTOMATIQUE SI FOND BLANC
    if img_array.mean() > 127:  # Fond majoritairement blanc
        img_array = 255 - img_array  # Inversion : chiffre blanc, fond noir

    # 3. Normaliser [0, 1]
    img_array_norm = img_array / 255.0
    x = img_array_norm.flatten().reshape(1, -1)

    # 4. DBN
    h1 = rbm1.transform(x)
    h2 = rbm2.transform(h1)
    pred = int(clf.predict(h2)[0])
    prob = clf.predict_proba(h2)[0]
    confidence = prob[pred]

    # 5. Retourner image traitée pour affichage
    img_display = img_array.astype('uint8')  # Pour affichage (non normalisé)

    return pred, confidence, prob, img_display

# === INTERFACE ===
uploaded_file = st.file_uploader("Choisis une image...", type=["png", "jpg", "jpeg", "bmp", "tiff"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Image originale", width=180, use_column_width=True)
    with col2:
        st.empty()  # Espace pour image traitée

    if st.button("Prédire le chiffre", type="primary"):
        with st.spinner("Analyse en cours..."):
            pred, confidence, all_probs, processed_img = preprocess_and_predict(image)

        # Afficher image traitée
        with col2:
            st.image(processed_img, caption="Image traitée (MNIST-style)", width=180, use_column_width=True)

        # Résultat
        st.success(f"**Prédiction : {pred}**")
        st.metric("Confiance", f"{confidence:.1%}")

        # Graphique des probabilités
        probs_dict = {f"{i}": float(all_probs[i]) for i in range(10)}
        st.bar_chart(probs_dict)

        # Message bonus
        if confidence > 0.9:
            st.balloons()
