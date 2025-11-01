# app.py
import streamlit as st
import joblib
import numpy as np
from PIL import Image
import os

# === CONFIGURATION ===
st.set_page_config(page_title="DBN MNIST", layout="centered")
st.title("Reconnaissance de Chiffres Manuscrits")
st.write("Upload une image (28x28, noir sur blanc) → prédiction instantanée !")

# === CHARGER LES MODÈLES ===
@st.cache_resource
def load_models():
    rbm1 = joblib.load('models/rbm1.pkl')
    rbm2 = joblib.load('models/rbm2.pkl')
    clf = joblib.load('models/classifier.pkl')
    return rbm1, rbm2, clf

rbm1, rbm2, clf = load_models()

# === FONCTION DE PRÉDICTION ===
def predict_image(image):
    # Redimensionner + gris + normaliser
    img = image.convert('L')
    img = img.resize((28, 28))
    img_array = np.array(img, dtype=np.float32) / 255.0
    x = img_array.flatten().reshape(1, -1)

    # DBN
    h1 = rbm1.transform(x)
    h2 = rbm2.transform(h1)
    pred = int(clf.predict(h2)[0])
    prob = clf.predict_proba(h2)[0]
    confidence = prob[pred]

    return pred, confidence, prob

# === INTERFACE ===
uploaded_file = st.file_uploader("Choisis une image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Image uploadée", width=200)

    if st.button("Prédire"):
        with st.spinner("Analyse en cours..."):
            pred, confidence, all_probs = predict_image(image)

        st.success(f"**Prédiction : {pred}**")
        st.metric("Confiance", f"{confidence:.1%}")

        # Bar chart des probabilités
        probs_df = {f"Chiffre {i}": all_probs[i] for i in range(10)}
        st.bar_chart(probs_df)