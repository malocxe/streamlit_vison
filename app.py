import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.set_page_config(page_title="Simulation Algorithme de Canny", layout="wide")

st.title("🖼️ Simulation de l'Algorithme de Canny")

st.write("""
Cette application montre toutes les étapes de l’algorithme de Canny :
1. Image originale
2. Filtrage gaussien
3. Calcul du gradient
4. Suppression des non-maxima
5. Double seuillage et hystérésis
6. Résultat final
""")

# Upload image
uploaded_file = st.file_uploader(
    "Télécharger une image",
    type=["jpg", "jpeg", "png"]
)

# Paramètres
st.sidebar.header("⚙️ Paramètres")

blur_size = st.sidebar.slider(
    "Taille du filtre gaussien",
    3,
    15,
    5,
    step=2
)

low_threshold = st.sidebar.slider(
    "Seuil bas",
    0,
    255,
    50
)

high_threshold = st.sidebar.slider(
    "Seuil haut",
    0,
    255,
    150
)

if uploaded_file is not None:

    # Charger image
    image = Image.open(uploaded_file)
    image_np = np.array(image)

    # Conversion RGB -> BGR pour OpenCV
    if len(image_np.shape) == 3:
        image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    else:
        image_cv = image_np

    # Niveaux de gris
    gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)

    # Étape 1 : Flou gaussien
    blurred = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)

    # Étape 2 : Gradient Sobel
    sobel_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)

    magnitude = cv2.magnitude(sobel_x, sobel_y)

    magnitude = np.uint8(
        255 * magnitude / np.max(magnitude)
    )

    # Direction du gradient
    angle = cv2.phase(sobel_x, sobel_y, angleInDegrees=True)

    # Étape 3, 4 et 5 : Canny complet
    canny = cv2.Canny(
        blurred,
        low_threshold,
        high_threshold
    )

    # Affichage
    st.header("📷 Résultats")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("1. Image Originale")
        st.image(image_np, use_container_width=True)

    with col2:
        st.subheader("2. Niveaux de gris")
        st.image(gray, clamp=True, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        st.subheader("3. Filtrage Gaussien")
        st.image(blurred, clamp=True, use_container_width=True)

    with col4:
        st.subheader("4. Gradient Magnitude")
        st.image(magnitude, clamp=True, use_container_width=True)

    col5, col6 = st.columns(2)

    with col5:
        st.subheader("5. Direction du Gradient")
        st.image(angle, clamp=True, use_container_width=True)

    with col6:
        st.subheader("6. Résultat Final - Canny")
        st.image(canny, clamp=True, use_container_width=True)

    st.success("Simulation terminée ✅")
