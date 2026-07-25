import os

import numpy as np
import streamlit as st
from PIL import Image
from tensorflow import keras

try:
    from groq import Groq
except ImportError:
    Groq = None


MODEL_PATH = "model/garbage_efficientnetb0.keras"
IMG_SIZE = (224, 224)

CLASS_NAMES = [
    "battery",
    "biological",
    "brown-glass",
    "cardboard",
    "green-glass",
    "metal",
    "paper",
    "plastic",
    "trash",
    "white-glass",
]

CLASS_ICONS = {
    "battery": "🔋",
    "biological": "🌿",
    "brown-glass": "🟤",
    "cardboard": "📦",
    "green-glass": "🟢",
    "metal": "⚙️",
    "paper": "📄",
    "plastic": "🧴",
    "trash": "🗑️",
    "white-glass": "⚪",
}

CLASS_TIPS = {
    "battery": (
        "Take batteries to a certified battery recycling or hazardous-waste "
        "collection point. Never place them in general waste because damaged "
        "batteries may leak chemicals or cause fires."
    ),
    "biological": (
        "Place food and organic waste in a compost or biological-waste bin. "
        "Remove plastic packaging before disposal."
    ),
    "brown-glass": (
        "Empty and rinse the container, then place it in the brown-glass "
        "recycling bin. Remove lids when required by local rules."
    ),
    "cardboard": (
        "Flatten the cardboard and place it in the paper or cardboard recycling "
        "bin. Keep it dry and remove food contamination."
    ),
    "green-glass": (
        "Empty and rinse the container, then place it in the green-glass "
        "recycling bin. Do not include ceramics or mirrors."
    ),
    "metal": (
        "Rinse metal cans and containers, then place them in the metal or mixed "
        "recycling bin. Sharp metal should be handled carefully."
    ),
    "paper": (
        "Place clean and dry paper in the paper recycling bin. Greasy, wet, or "
        "heavily contaminated paper may need to go into general waste."
    ),
    "plastic": (
        "Check the recycling symbol and local rules. Rinse the item and place "
        "accepted plastic in the recycling bin."
    ),
    "trash": (
        "Place non-recyclable waste in the general-waste bin. Reuse the item "
        "first when possible."
    ),
    "white-glass": (
        "Empty and rinse the container, then place it in the clear or white-glass "
        "recycling bin. Do not include window glass, mirrors, or ceramics."
    ),
}


st.set_page_config(
    page_title="EcoScan AI",
    page_icon="♻️",
    layout="wide",
)

st.markdown(
    """
    <style>
        .main-title {
            font-size: 3rem;
            font-weight: 800;
            text-align: center;
            margin-bottom: 0;
        }

        .subtitle {
            text-align: center;
            color: #777;
            margin-bottom: 2rem;
        }

        .result-card {
            padding: 1.5rem;
            border-radius: 15px;
            border: 1px solid rgba(120, 120, 120, 0.25);
            margin-top: 1rem;
        }

        .prediction-name {
            font-size: 2rem;
            font-weight: 800;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None

    return keras.models.load_model(MODEL_PATH)


def prepare_image(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB")
    image = image.resize(IMG_SIZE)

    image_array = np.asarray(image, dtype=np.float32)
    image_array = np.expand_dims(image_array, axis=0)

    return image_array


def predict_image(model, image: Image.Image):
    image_array = prepare_image(image)

    probabilities = model.predict(image_array, verbose=0)[0]

    class_index = int(np.argmax(probabilities))
    confidence = float(probabilities[class_index])

    return CLASS_NAMES[class_index], confidence, probabilities


def get_ai_guide(label: str, fallback_tip: str) -> str:
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key or Groq is None:
        return fallback_tip

    prompt = f"""
You are a smart waste disposal assistant.

The image classification model identified the waste as: {label}

Give a short and practical disposal guide.

Include:
- Correct disposal bin
- Whether the material is recyclable, compostable, reusable, or general waste
- Any important safety advice
"""

    try:
        client = Groq(api_key=api_key)

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            temperature=0.25,
            max_tokens=250,
        )

        return response.choices[0].message.content

    except Exception:
        return fallback_tip


st.markdown(
    '<h1 class="main-title">♻️ EcoScan AI</h1>',
    unsafe_allow_html=True,
)

st.markdown(
    '<p class="subtitle">Smart waste classification and recycling guidance</p>',
    unsafe_allow_html=True,
)

model = load_model()

if model is None:
    st.warning(
        "The trained model is not available yet. "
        "Upload it to `model/garbage_efficientnetb0.keras`."
    )

uploaded_file = st.file_uploader(
    "Upload a waste image",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    left_column, right_column = st.columns(2)

    with left_column:
        st.subheader("Uploaded Image")
        st.image(image, use_container_width=True)

    with right_column:
        st.subheader("Classification Result")

        if model is not None:
            with st.spinner("Analyzing the image..."):
                label, confidence, probabilities = predict_image(model, image)

            icon = CLASS_ICONS.get(label, "♻️")
            readable_label = label.replace("-", " ").title()

            st.markdown(
                f"""
                <div class="result-card">
                    <div class="prediction-name">
                        {icon} {readable_label}
                    </div>
                    <p><strong>Confidence:</strong> {confidence * 100:.2f}%</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.progress(float(confidence))

            st.subheader("Disposal Guidance")

            guide = get_ai_guide(
                label,
                CLASS_TIPS[label],
            )

            st.info(guide)

            st.subheader("Class Probabilities")

            probability_data = {
                class_name.replace("-", " ").title(): float(probability)
                for class_name, probability in zip(
                    CLASS_NAMES,
                    probabilities,
                )
            }

            st.bar_chart(probability_data)

st.divider()

st.caption(
    "EcoScan AI uses an EfficientNetB0 image-classification model. "
    "Disposal rules may vary by location."
)
