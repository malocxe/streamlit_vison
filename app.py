import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageOps


st.set_page_config(
    page_title="Simulation de l'algorithme de Canny",
    page_icon=":frame_with_picture:",
    layout="wide",
)


def load_uploaded_image(uploaded_file):
    """Load any supported image as RGB uint8, with EXIF orientation applied."""
    image = Image.open(uploaded_file)
    image = ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    return np.array(image, dtype=np.uint8)


def resize_if_needed(rgb_image, max_side):
    height, width = rgb_image.shape[:2]
    largest_side = max(height, width)

    if largest_side <= max_side:
        return rgb_image, False

    scale = max_side / largest_side
    new_width = max(5, int(width * scale))
    new_height = max(5, int(height * scale))
    resized = cv2.resize(rgb_image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return resized, True


def normalize_to_uint8(values):
    values = np.asarray(values, dtype=np.float32)
    max_value = float(values.max()) if values.size else 0.0

    if max_value <= 0:
        return np.zeros(values.shape, dtype=np.uint8)

    return np.clip(values * 255.0 / max_value, 0, 255).astype(np.uint8)


def non_maximum_suppression(magnitude, angle):
    height, width = magnitude.shape
    suppressed = np.zeros((height, width), dtype=np.float32)
    angle = angle % 180

    for y in range(1, height - 1):
        for x in range(1, width - 1):
            direction = angle[y, x]
            current = magnitude[y, x]

            if (0 <= direction < 22.5) or (157.5 <= direction <= 180):
                before, after = magnitude[y, x - 1], magnitude[y, x + 1]
            elif 22.5 <= direction < 67.5:
                before, after = magnitude[y - 1, x + 1], magnitude[y + 1, x - 1]
            elif 67.5 <= direction < 112.5:
                before, after = magnitude[y - 1, x], magnitude[y + 1, x]
            else:
                before, after = magnitude[y - 1, x - 1], magnitude[y + 1, x + 1]

            if current >= before and current >= after:
                suppressed[y, x] = current

    return suppressed


def double_threshold(image, low_threshold, high_threshold):
    strong_value = 255
    weak_value = 75

    thresholded = np.zeros(image.shape, dtype=np.uint8)
    thresholded[image >= high_threshold] = strong_value
    thresholded[(image >= low_threshold) & (image < high_threshold)] = weak_value

    return thresholded, weak_value, strong_value


def hysteresis(thresholded, weak_value=75, strong_value=255):
    candidate_edges = (thresholded > 0).astype(np.uint8)
    component_count, labels = cv2.connectedComponents(candidate_edges, connectivity=8)
    result = np.zeros(thresholded.shape, dtype=np.uint8)

    for label in range(1, component_count):
        component = labels == label
        if np.any(thresholded[component] == strong_value):
            result[component] = strong_value

    return result


@st.cache_data(show_spinner=False)
def build_canny_steps(rgb_image, blur_size, low_threshold, high_threshold):
    gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)

    sobel_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)

    magnitude_float = cv2.magnitude(sobel_x, sobel_y)
    magnitude = normalize_to_uint8(magnitude_float)
    angle = cv2.phase(sobel_x, sobel_y, angleInDegrees=True)
    angle_display = normalize_to_uint8(angle)

    suppressed_float = non_maximum_suppression(magnitude_float, angle)
    suppressed = normalize_to_uint8(suppressed_float)

    thresholded, weak_value, strong_value = double_threshold(
        suppressed, low_threshold, high_threshold
    )
    hysteresis_result = hysteresis(thresholded, weak_value, strong_value)
    opencv_canny = cv2.Canny(blurred, low_threshold, high_threshold)

    return {
        "gray": gray,
        "blurred": blurred,
        "sobel_x": sobel_x,
        "sobel_y": sobel_y,
        "magnitude_float": magnitude_float,
        "magnitude": magnitude,
        "angle": angle,
        "angle_display": angle_display,
        "suppressed": suppressed,
        "thresholded": thresholded,
        "hysteresis": hysteresis_result,
        "opencv_canny": opencv_canny,
    }


def show_image(title, image, caption=None):
    st.subheader(title)
    st.image(image, caption=caption, clamp=True, use_container_width=True)


st.title("Simulation de l'algorithme de Canny")

st.markdown(
    """
Cette application montre le chemin complet d'une image vers ses contours :
gris, flou gaussien, gradients Sobel, suppression des non-maxima, double
seuillage, hysteresis et comparaison avec `cv2.Canny`.
"""
)

with st.sidebar:
    st.header("Parametres")

    blur_size = st.slider(
        "Taille du filtre gaussien",
        min_value=3,
        max_value=15,
        value=5,
        step=2,
        help="La taille doit etre impaire. Plus elle est grande, plus le bruit est lisse.",
    )

    low_threshold = st.slider("Seuil bas", 0, 255, 50)
    high_threshold = st.slider("Seuil haut", 0, 255, 150)

    max_processing_side = st.slider(
        "Taille max traitee",
        min_value=300,
        max_value=1600,
        value=900,
        step=100,
        help="Reduit les grandes images pour garder une simulation fluide.",
    )

    if low_threshold >= high_threshold:
        st.warning("Le seuil bas doit rester inferieur au seuil haut.")

uploaded_file = st.file_uploader(
    "Choisir une image",
    type=["jpg", "jpeg", "png", "bmp", "webp"],
)

if uploaded_file is None:
    st.info("Ajoute une image pour lancer la simulation.")
    st.stop()

if low_threshold >= high_threshold:
    st.stop()

try:
    rgb_image = load_uploaded_image(uploaded_file)
except Exception as exc:
    st.error(f"Impossible de lire cette image : {exc}")
    st.stop()

height, width = rgb_image.shape[:2]
if height < 5 or width < 5:
    st.error("L'image est trop petite. Utilise une image d'au moins 5 x 5 pixels.")
    st.stop()

rgb_image, was_resized = resize_if_needed(rgb_image, max_processing_side)
height, width = rgb_image.shape[:2]

if was_resized:
    st.caption(
        f"Image redimensionnee pour le calcul : {width} x {height} pixels."
    )

steps = build_canny_steps(rgb_image, blur_size, low_threshold, high_threshold)

st.header("Resultats")

first_row = st.columns(3)
with first_row[0]:
    show_image("1. Image originale", rgb_image)
with first_row[1]:
    show_image("2. Niveaux de gris", steps["gray"])
with first_row[2]:
    show_image("3. Filtrage gaussien", steps["blurred"])

second_row = st.columns(3)
with second_row[0]:
    show_image("4. Magnitude du gradient", steps["magnitude"])
with second_row[1]:
    show_image("5. Direction du gradient", steps["angle_display"])
with second_row[2]:
    show_image("6. Suppression des non-maxima", steps["suppressed"])

third_row = st.columns(3)
with third_row[0]:
    show_image("7. Double seuillage", steps["thresholded"])
with third_row[1]:
    show_image("8. Hysteresis simulee", steps["hysteresis"])
with third_row[2]:
    show_image("9. resultat final ", steps["opencv_canny"])

st.header("Simulation matricielle locale")
st.write(
    "Selectionne une petite zone de l'image filtree pour voir le calcul Sobel "
    "sur une matrice 3 x 3."
)

max_x = max(width - 5, 0)
max_y = max(height - 5, 0)

position_cols = st.columns(2)
with position_cols[0]:
    x_pos = st.slider("Position X", 0, max_x, min(max_x, width // 2))
with position_cols[1]:
    y_pos = st.slider("Position Y", 0, max_y, min(max_y, height // 2))

sample = steps["blurred"][y_pos : y_pos + 5, x_pos : x_pos + 5]
region = sample[1:4, 1:4]

matrix_cols = st.columns(2)
with matrix_cols[0]:
    st.subheader("Matrice 5 x 5 extraite")
    st.dataframe(sample, use_container_width=True)

with matrix_cols[1]:
    gaussian_kernel_1d = cv2.getGaussianKernel(blur_size, 0)
    gaussian_kernel = gaussian_kernel_1d @ gaussian_kernel_1d.T
    gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()
    st.subheader("Filtre gaussien")
    st.dataframe(np.round(gaussian_kernel, 4), use_container_width=True)

sobel_kernel_x = np.array(
    [
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1],
    ]
)
sobel_kernel_y = np.array(
    [
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1],
    ]
)

sobel_cols = st.columns(3)
with sobel_cols[0]:
    st.subheader("Region 3 x 3")
    st.dataframe(region, use_container_width=True)
with sobel_cols[1]:
    st.subheader("Masque Sobel X")
    st.dataframe(sobel_kernel_x, use_container_width=True)
with sobel_cols[2]:
    st.subheader("Masque Sobel Y")
    st.dataframe(sobel_kernel_y, use_container_width=True)

calc_x = region * sobel_kernel_x
calc_y = region * sobel_kernel_y
sum_x = int(calc_x.sum())
sum_y = int(calc_y.sum())
magnitude_manual = float(np.sqrt(sum_x**2 + sum_y**2))
angle_manual = float(np.degrees(np.arctan2(sum_y, sum_x)))

calc_cols = st.columns(2)
with calc_cols[0]:
    st.subheader("Calcul de Gx")
    st.dataframe(calc_x, use_container_width=True)
    st.write(f"Gx = {sum_x}")
with calc_cols[1]:
    st.subheader("Calcul de Gy")
    st.dataframe(calc_y, use_container_width=True)
    st.write(f"Gy = {sum_y}")

metric_cols = st.columns(2)
metric_cols[0].metric("Magnitude locale", f"{magnitude_manual:.2f}")
metric_cols[1].metric("Angle local", f"{angle_manual:.2f} deg")

local_threshold = max(float(steps["magnitude"].mean()) * 0.3, 1.0)
if magnitude_manual > local_threshold * 2:
    st.success("Interpretation : contour fort detecte.")
elif magnitude_manual > local_threshold:
    st.warning("Interpretation : contour faible detecte.")
else:
    st.error("Interpretation : pas de contour detecte.")

st.subheader("Zone selectionnee")
zoom = cv2.cvtColor(steps["blurred"].copy(), cv2.COLOR_GRAY2RGB)
cv2.rectangle(zoom, (x_pos, y_pos), (x_pos + 5, y_pos + 5), (0, 255, 0), 2)
st.image(zoom, use_container_width=True)
