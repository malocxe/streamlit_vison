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
    # Conversion RGB -> BGR pour OpenCV
    image_np = np.array(image)
    
    if len(image_np.shape) == 3:
        image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    else:
        image_cv = image_np
    
    # ✅ SAFE GRAYSCALE CONVERSION
    if len(image_cv.shape) == 3:
        gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
    else:
        gray = image_cv

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

        # =====================================================
    # SIMULATION MATRICIELLE
    # =====================================================

    st.header("🧮 Simulation Matricielle Pas à Pas")

    st.write("""
    Cette partie montre comment l’algorithme travaille
    directement sur les matrices de pixels.
    """)

    # =====================================================
    # CHOIX DYNAMIQUE D'UNE REGION
    # =====================================================

    st.subheader("📍 Sélection d'une région de l'image")

    h, w = blurred.shape

    x_pos = st.slider(
        "Position X",
        0,
        w - 6,
        0
    )

    y_pos = st.slider(
        "Position Y",
        0,
        h - 6,
        0
    )

    # Région 5x5
    sample = blurred[y_pos:y_pos+5, x_pos:x_pos+5]

    st.subheader("1️⃣ Matrice 5x5 extraite")

    st.dataframe(sample)

    # =====================================================
    # FILTRE GAUSSIEN
    # =====================================================

    st.subheader("2️⃣ Filtre Gaussien Généré")

    gaussian_kernel_1d = cv2.getGaussianKernel(blur_size, 0)

    gaussian_kernel = gaussian_kernel_1d @ gaussian_kernel_1d.T

    gaussian_kernel = gaussian_kernel / np.sum(gaussian_kernel)

    st.write("Matrice du filtre gaussien :")

    st.dataframe(
        np.round(gaussian_kernel, 4)
    )

    # =====================================================
    # SOBEL
    # =====================================================

    sobel_kernel_x = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ])

    sobel_kernel_y = np.array([
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]
    ])

    st.subheader("3️⃣ Masque Sobel X")

    st.dataframe(sobel_kernel_x)

    st.subheader("4️⃣ Masque Sobel Y")

    st.dataframe(sobel_kernel_y)

    # =====================================================
    # REGION 3x3
    # =====================================================

    region = sample[1:4, 1:4]

    st.subheader("5️⃣ Région 3x3 analysée")

    st.dataframe(region)

    # =====================================================
    # CALCUL Gx
    # =====================================================

    calc_x = region * sobel_kernel_x

    sum_x = np.sum(calc_x)

    st.subheader("6️⃣ Calcul du Gradient Horizontal Gx")

    st.write("Multiplication élément par élément :")

    st.dataframe(calc_x)

    st.write(f"Gx = {sum_x}")

    # =====================================================
    # CALCUL Gy
    # =====================================================

    calc_y = region * sobel_kernel_y

    sum_y = np.sum(calc_y)

    st.subheader("7️⃣ Calcul du Gradient Vertical Gy")

    st.write("Multiplication élément par élément :")

    st.dataframe(calc_y)

    st.write(f"Gy = {sum_y}")

    # =====================================================
    # MAGNITUDE
    # =====================================================

    magnitude_manual = np.sqrt(sum_x**2 + sum_y**2)

    st.subheader("8️⃣ Magnitude du Gradient")

    st.latex(r"G = \sqrt{G_x^2 + G_y^2}")

    st.write(
        f"G = sqrt({sum_x}² + {sum_y}²)"
    )

    st.write(
        f"Magnitude = {magnitude_manual:.2f}"
    )

    # =====================================================
    # ANGLE
    # =====================================================

    angle_manual = np.arctan2(sum_y, sum_x) * 180 / np.pi

    st.subheader("9️⃣ Direction du Gradient")

    st.latex(r"\theta = tan^{-1}(G_y/G_x)")

    st.write(
        f"Angle = {angle_manual:.2f}°"
    )

    # =====================================================
    # INTERPRETATION
    # =====================================================

    st.subheader("🔍 Interprétation")
    
    local_threshold = np.mean(magnitude) * 0.3  # seuil adaptatif
    
    if magnitude_manual > local_threshold * 2:
        st.success("Contour FORT détecté")
    
    elif magnitude_manual > local_threshold:
        st.warning("Contour FAIBLE détecté")
    
    else:
        st.error("Pas de contour détecté")
    # =====================================================
    # VISUALISATION ZONE
    # =====================================================

    st.subheader("🖼️ Zone sélectionnée dans l'image")

    zoom = cv2.rectangle(
        cv2.cvtColor(blurred.copy(), cv2.COLOR_GRAY2BGR),
        (x_pos, y_pos),
        (x_pos + 5, y_pos + 5),
        (0, 255, 0),
        2
    )

    st.image(
        zoom,
        use_container_width=True
    )

    st.success("Simulation terminée ✅")
