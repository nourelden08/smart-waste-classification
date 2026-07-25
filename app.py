import os
from pathlib import Path
from textwrap import dedent

import numpy as np
import streamlit as st
from PIL import Image
from tensorflow import keras

from dotenv import load_dotenv

try:
    from groq import Groq
except ImportError:
    Groq = None

load_dotenv()

APP_TITLE = "EcoScan AI"
MODEL_PATH = Path("model/garbage_efficientnetb0.keras")
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
        "collection point. Never place damaged batteries in general waste."
    ),
    "biological": (
        "Place food and organic waste in a compost or biological-waste bin. "
        "Remove plastic packaging first."
    ),
    "brown-glass": (
        "Empty and rinse the container, then place it in the brown-glass "
        "recycling bin."
    ),
    "cardboard": (
        "Flatten clean and dry cardboard, then place it in the paper or "
        "cardboard recycling bin."
    ),
    "green-glass": (
        "Empty and rinse the container, then place it in the green-glass "
        "recycling bin."
    ),
    "metal": (
        "Rinse metal cans and containers, then place them in the metal or "
        "mixed-recycling bin."
    ),
    "paper": (
        "Place clean and dry paper in the paper recycling bin. Wet or greasy "
        "paper may not be recyclable."
    ),
    "plastic": (
        "Check the recycling symbol and local rules. Rinse accepted plastic "
        "before recycling."
    ),
    "trash": (
        "Place non-recyclable waste in the general-waste bin. Reuse the item "
        "first when possible."
    ),
    "white-glass": (
        "Empty and rinse the container, then place it in the clear or "
        "white-glass recycling bin."
    ),
}


st.set_page_config(
    page_title=APP_TITLE,
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def render_html(content: str) -> None:
    st.html(dedent(content))


render_html(
    """
    <style>
        :root {
            --green: #00e8a2;
            --green-dark: #00b87a;
            --panel: rgba(18, 23, 31, 0.94);
            --muted: #9da8b7;
            --border: rgba(255, 255, 255, 0.10);
        }

        .stApp {
            background:
                radial-gradient(
                    circle at 15% 15%,
                    rgba(0, 232, 162, 0.12),
                    transparent 30%
                ),
                radial-gradient(
                    circle at 85% 15%,
                    rgba(39, 120, 255, 0.10),
                    transparent 28%
                ),
                #0b0f14;
        }

        [data-testid="stHeader"] {
            background: transparent;
        }

        [data-testid="stSidebar"] {
            display: none;
        }

        .block-container {
            padding-top: 1rem;
            padding-bottom: 3rem;
            max-width: 1220px;
        }

        .eco-container {
            max-width: 1180px;
            margin: 0 auto;
            padding: 0.5rem 0 2rem;
        }

        .topbar {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0.8rem 0 1.2rem;
        }

        .brand {
            display: flex;
            align-items: center;
            gap: 0.8rem;
        }

        .logo {
            width: 44px;
            height: 44px;
            border-radius: 14px;
            display: flex;
            align-items: center;
            justify-content: center;
            background: linear-gradient(
                135deg,
                var(--green),
                var(--green-dark)
            );
            color: #04100c;
            font-size: 1.45rem;
            font-weight: 900;
            box-shadow: 0 12px 30px rgba(0, 232, 162, 0.20);
        }

        .brand-title {
            font-size: 1.45rem;
            font-weight: 900;
            letter-spacing: -0.02em;
        }

        .brand-title span {
            color: var(--green);
        }

        .pill {
            border: 1px solid var(--border);
            background: rgba(255, 255, 255, 0.04);
            border-radius: 999px;
            padding: 0.5rem 0.8rem;
            color: #c9d2dd;
            font-size: 0.86rem;
        }

        .live-dot {
            display: inline-block;
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: var(--green);
            margin-right: 0.45rem;
            box-shadow: 0 0 15px rgba(0, 232, 162, 0.9);
        }

        .hero {
            text-align: center;
            padding: 4.5rem 1rem 2.5rem;
        }

        .eyebrow {
            display: inline-block;
            color: var(--green);
            border: 1px solid rgba(0, 232, 162, 0.28);
            background: rgba(0, 232, 162, 0.07);
            border-radius: 999px;
            padding: 0.45rem 0.75rem;
            font-size: 0.82rem;
            font-weight: 800;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }

        .hero h1 {
            font-size: clamp(3rem, 8vw, 6.7rem);
            line-height: 0.95;
            margin: 1.3rem 0 1rem;
            letter-spacing: -0.065em;
            font-weight: 950;
        }

        .hero h1 span {
            color: var(--green);
        }

        .hero p {
            max-width: 760px;
            margin: 0 auto;
            color: var(--muted);
            font-size: 1.12rem;
            line-height: 1.75;
        }

        .feature-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 1rem;
            margin-top: 3rem;
        }

        .feature-card {
            border: 1px solid var(--border);
            background: var(--panel);
            border-radius: 22px;
            padding: 1.4rem;
            min-height: 175px;
        }

        .feature-card h3 {
            margin: 0.85rem 0 0.4rem;
        }

        .feature-card p {
            color: var(--muted);
            line-height: 1.6;
            font-size: 0.95rem;
        }

        .feature-icon {
            font-size: 1.8rem;
        }

        .page-title {
            margin: 1rem 0 1.5rem;
        }

        .page-title h2 {
            font-size: 2.5rem;
            margin: 0.5rem 0;
            letter-spacing: -0.035em;
        }

        .page-title p {
            color: var(--muted);
            max-width: 760px;
        }

        .result-card {
            border: 1px solid var(--border);
            background: var(--panel);
            border-radius: 20px;
            padding: 1.4rem;
            margin-bottom: 1rem;
        }

        .prediction-name {
            font-size: 2rem;
            font-weight: 900;
            margin-bottom: 0.25rem;
        }

        .small-muted {
            color: var(--muted);
        }

        .footer {
            text-align: center;
            color: #6f7b89;
            font-size: 0.85rem;
            padding-top: 2rem;
        }

        div.stButton > button {
            border-radius: 999px;
            border: 1px solid rgba(0, 232, 162, 0.35);
            background: linear-gradient(
                135deg,
                #00e8a2,
                #00b87a
            );
            color: #04100c;
            font-weight: 900;
            min-height: 46px;
            padding: 0 1.3rem;
        }

        div.stButton > button:hover {
            border-color: #00e8a2;
            transform: translateY(-1px);
        }

        [data-testid="stFileUploader"] {
            border: 1px solid var(--border);
            background: var(--panel);
            border-radius: 18px;
            padding: 1rem;
        }

        [data-testid="stChatMessage"] {
            border: 1px solid var(--border);
            border-radius: 16px;
            background: rgba(255, 255, 255, 0.025);
            padding: 0.6rem;
        }

        @media (max-width: 900px) {
            .feature-grid {
                grid-template-columns: 1fr;
            }

            .hero {
                padding-top: 3.5rem;
            }

            .topbar {
                gap: 1rem;
                align-items: flex-start;
            }

            .pill {
                font-size: 0.75rem;
            }
        }
    </style>
    """
)


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None

    return keras.models.load_model(
        MODEL_PATH,
        compile=False,
    )


def prepare_image(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB")
    image = image.resize(IMG_SIZE)

    image_array = np.asarray(
        image,
        dtype=np.float32,
    )

    return np.expand_dims(
        image_array,
        axis=0,
    )


def predict_image(model, image: Image.Image):
    probabilities = model.predict(
        prepare_image(image),
        verbose=0,
    )[0]

    class_index = int(
        np.argmax(probabilities)
    )

    confidence = float(
        probabilities[class_index]
    )

    return (
        CLASS_NAMES[class_index],
        confidence,
        probabilities,
    )


def ask_groq(
    prompt: str,
    context_label: str | None = None,
) -> str:
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key or Groq is None:
        return (
            "The live chatbot is optional. Add a GROQ_API_KEY "
            "environment variable to enable AI answers. The built-in "
            "disposal guidance still works without it."
        )

    system_prompt = (
        "You are EcoScan AI, a concise recycling and waste-disposal "
        "assistant. Give practical and safe advice. Mention that local "
        "recycling rules may vary."
    )

    if context_label:
        system_prompt += (
            f" The latest model prediction is: {context_label}."
        )

    try:
        client = Groq(
            api_key=api_key
        )

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            temperature=0.25,
            max_tokens=350,
        )

        return response.choices[0].message.content

    except Exception as error:
        return (
            "The chatbot is unavailable right now. "
            f"Details: {error}"
        )


def initialize_state():
    if "page" not in st.session_state:
        st.session_state.page = "landing"

    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []

    if "last_label" not in st.session_state:
        st.session_state.last_label = None


def render_topbar(status_text: str):
    render_html(
        f"""
        <div class="eco-container">
            <div class="topbar">
                <div class="brand">
                    <div class="logo">♻</div>

                    <div class="brand-title">
                        EcoScan<span>AI</span>
                    </div>
                </div>

                <div class="pill">
                    <span class="live-dot"></span>
                    {status_text}
                </div>
            </div>
        </div>
        """
    )


def render_landing_page():
    render_topbar(
        "EfficientNetB0 Powered"
    )

    render_html(
        """
        <div class="eco-container">
            <section class="hero">
                <div class="eyebrow">
                    Smart waste intelligence
                </div>

                <h1>
                    Scan waste.<br>
                    <span>Dispose smarter.</span>
                </h1>

                <p>
                    EcoScan AI classifies waste across 10 categories
                    using a fine-tuned EfficientNetB0 model, then
                    provides practical recycling guidance and an
                    optional AI-powered assistant.
                </p>
            </section>
        </div>
        """
    )

    left_column, center_column, right_column = st.columns(
        [1.7, 1, 1.7]
    )

    with center_column:
        if st.button(
            "Launch EcoScan",
            use_container_width=True,
        ):
            st.session_state.page = "scanner"
            st.rerun()

    render_html(
        """
        <div class="eco-container">
            <div class="feature-grid">

                <div class="feature-card">
                    <div class="feature-icon">📷</div>

                    <h3>Image Classification</h3>

                    <p>
                        Upload a waste image and classify it into
                        one of 10 supported categories.
                    </p>
                </div>

                <div class="feature-card">
                    <div class="feature-icon">📊</div>

                    <h3>Confidence Dashboard</h3>

                    <p>
                        Review the predicted class, confidence level,
                        and probability distribution.
                    </p>
                </div>

                <div class="feature-card">
                    <div class="feature-icon">💬</div>

                    <h3>Recycling Assistant</h3>

                    <p>
                        Ask disposal and recycling questions through
                        the optional Groq-powered chatbot.
                    </p>
                </div>

            </div>

            <div class="footer">
                EcoScan AI · Intelligent waste classification
                and recycling guidance
            </div>
        </div>
        """
    )


def render_scanner_page():
    render_topbar(
        "Live Inference"
    )

    back_column, spacer_column, clear_column = st.columns(
        [1, 6, 1.4]
    )

    with back_column:
        if st.button("← Back"):
            st.session_state.page = "landing"
            st.rerun()

    with clear_column:
        if st.button("Clear Chat"):
            st.session_state.chat_messages = []
            st.rerun()

    render_html(
        """
        <div class="page-title">
            <div class="eyebrow">
                Classifier + Disposal Guide + Chat
            </div>

            <h2>Waste Classification</h2>

            <p>
                Upload an image, inspect the prediction, and ask
                the recycling assistant follow-up questions.
            </p>
        </div>
        """
    )

    model = load_model()

    if model is None:
        st.error(
            "Model file not found. Place "
            "`garbage_efficientnetb0.keras` inside the "
            "`model` folder."
        )
        st.stop()

    upload_column, result_column = st.columns(
        [1, 1]
    )

    with upload_column:
        st.subheader(
            "Upload Waste Image"
        )

        uploaded_file = st.file_uploader(
            "Choose a JPG, JPEG, or PNG image",
            type=["jpg", "jpeg", "png"],
        )

        image = None

        if uploaded_file is not None:
            try:
                image = Image.open(
                    uploaded_file
                )

                st.image(
                    image,
                    use_container_width=True,
                )

            except Exception as error:
                st.error(
                    f"Could not open the image: {error}"
                )

    with result_column:
        st.subheader(
            "Classification Result"
        )

        if image is None:
            st.info(
                "Upload an image to start classification."
            )

        else:
            with st.spinner(
                "Analyzing image..."
            ):
                (
                    label,
                    confidence,
                    probabilities,
                ) = predict_image(
                    model,
                    image,
                )

            st.session_state.last_label = label

            icon = CLASS_ICONS.get(
                label,
                "♻️",
            )

            readable_label = (
                label
                .replace("-", " ")
                .title()
            )

            render_html(
                f"""
                <div class="result-card">
                    <div class="prediction-name">
                        {icon} {readable_label}
                    </div>

                    <div class="small-muted">
                        Confidence:
                        {confidence * 100:.2f}%
                    </div>
                </div>
                """
            )

            st.progress(
                confidence
            )

            st.subheader(
                "Disposal Guidance"
            )

            st.info(
                CLASS_TIPS[label]
            )

            st.subheader(
                "Class Probabilities"
            )

            chart_data = {
                class_name
                .replace("-", " ")
                .title(): float(probability)
                for class_name, probability in zip(
                    CLASS_NAMES,
                    probabilities,
                )
            }

            st.bar_chart(
                chart_data
            )

    st.divider()

    st.subheader(
        "Recycling Assistant"
    )

    st.caption(
        "The chatbot requires a GROQ_API_KEY. "
        "The image classifier works without it."
    )

    for message in st.session_state.chat_messages:
        with st.chat_message(
            message["role"]
        ):
            st.markdown(
                message["content"]
            )

    user_prompt = st.chat_input(
        "Ask about recycling or disposal..."
    )

    if user_prompt:
        st.session_state.chat_messages.append(
            {
                "role": "user",
                "content": user_prompt,
            }
        )

        with st.chat_message("user"):
            st.markdown(
                user_prompt
            )

        with st.chat_message("assistant"):
            with st.spinner(
                "Thinking..."
            ):
                reply = ask_groq(
                    user_prompt,
                    st.session_state.last_label,
                )

            st.markdown(
                reply
            )

        st.session_state.chat_messages.append(
            {
                "role": "assistant",
                "content": reply,
            }
        )

    render_html(
        """
        <div class="footer">
            Disposal rules may vary by location.
            Check your local recycling authority.
        </div>
        """
    )


initialize_state()

if st.session_state.page == "landing":
    render_landing_page()
else:
    render_scanner_page()