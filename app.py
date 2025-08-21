import streamlit as st
import tensorflow as tf
from tensorflow.keras import layers
from PIL import Image
import numpy as np
import io


def build_generator():
    inputs = layers.Input(shape=(128, 128, 3))

    # Encoder
    down1 = layers.Conv2D(64, (4, 4), strides=2, padding='same')(inputs)
    down1 = layers.LeakyReLU()(down1)

    down2 = layers.Conv2D(128, (4, 4), strides=2, padding='same')(down1)
    down2 = layers.LeakyReLU()(down2)

    # Bottleneck
    bottleneck = layers.Conv2D(256, (4, 4), strides=2, padding='same')(down2)
    bottleneck = layers.ReLU()(bottleneck)

    # Decoder
    up1 = layers.Conv2DTranspose(128, (4, 4), strides=2, padding='same')(bottleneck)
    up1 = layers.ReLU()(up1)

    up2 = layers.Conv2DTranspose(64, (4, 4), strides=2, padding='same')(up1)
    up2 = layers.ReLU()(up2)

    outputs = layers.Conv2DTranspose(3, (4, 4), strides=2, padding='same', activation='tanh')(up2)

    return tf.keras.Model(inputs, outputs, name="generator")

# ------------------------------
# Load generator weights
# ------------------------------
generator = build_generator()
try:
    generator.load_weights("cycle_gan.weights.h5")   # replace with your file path
    model_loaded = True
    st.success("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"⚠️ Could not load model: {e}")
    model_loaded = False

# ------------------------------
# Preprocess & Postprocess
# ------------------------------
def preprocess_image(img):
    img = img.resize((128, 128))  # match training size
    img_array = np.array(img).astype("float32")
    img_array = (img_array / 127.5) - 1.0   # normalize to [-1,1]
    return np.expand_dims(img_array, axis=0)

def postprocess_image(pred):
    pred = (pred + 1.0) * 127.5   # back to [0,255]
    pred = np.clip(pred[0], 0, 255).astype(np.uint8)
    return Image.fromarray(pred)


st.title("🎨 CycleGAN Image Translator")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    input_image = Image.open(uploaded_file).convert("RGB")
    st.image(input_image, caption="📥 Uploaded Image", use_container_width=True)

    if st.button("Translate Image"):
        if model_loaded:
            # preprocess → predict → postprocess
            preprocessed = preprocess_image(input_image)
            prediction = generator(preprocessed, training=False)
            output_image = postprocess_image(prediction)

            col1, col2 = st.columns(2)
            with col1:
                st.image(input_image, caption="Original", use_container_width=True)
            with col2:
                st.image(output_image, caption="🎨 Translated", use_container_width=True)

            # download option
            buf = io.BytesIO()
            output_image.save(buf, format="PNG")
            st.download_button("Download Output", data=buf.getvalue(),
                               file_name="translated.png", mime="image/png")
        else:
            st.error("⚠️ Model not loaded.")