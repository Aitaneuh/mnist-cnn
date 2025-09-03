import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model import CNNModel
from streamlit_drawable_canvas import st_canvas

# ----------------------------
# Load model
# ----------------------------
model = CNNModel()
model.load_state_dict(torch.load("mnist_model.pth", map_location="cpu"))
model.eval()

# ----------------------------
# Streamlit UI
# ----------------------------
st.title("MNIST CNN")
st.write("Draw a digit or upload an image to see the CNN prediction.")

# ----------------------------
# Canvas for drawing
# ----------------------------
stroke_width = st.slider("Brush size", 5, 40, 10)
canvas_result = st_canvas(
    fill_color="black",
    stroke_width=stroke_width,
    stroke_color="white",
    background_color="black",
    width=280,
    height=280,
    drawing_mode="freedraw",
    key="canvas",
)


# ----------------------------
# Image upload
# ----------------------------
uploaded_file = st.file_uploader("Or upload a digit image", type=["png", "jpg", "jpeg"])

# ----------------------------
# Process image
# ----------------------------
def process_image(img: Image.Image):
    img = img.convert("L")        # grayscale
    img = img.resize((28, 28))    # resize
    img_array = np.array(img)/255.0
    img_tensor = torch.tensor(img_array, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    img_tensor = (img_tensor - 0.5)/0.5
    return img_array, img_tensor

img_array = None
img_tensor = None

if uploaded_file:
    img = Image.open(uploaded_file)
    img_array, img_tensor = process_image(img)

elif canvas_result.image_data is not None:
    # Convert RGBA canvas to PIL image
    img = Image.fromarray(canvas_result.image_data.astype("uint8")).convert("L")
    img_array, img_tensor = process_image(img)

# ----------------------------
# Prediction and visualization
# ----------------------------
if img_tensor is not None and img_array is not None:
    with torch.no_grad():
        output = model(img_tensor)
        probs = F.softmax(output, dim=1).cpu().numpy()[0]
        prediction = int(np.argmax(probs))
        feature_maps = F.relu(model.conv1(img_tensor)).squeeze(0)

    st.subheader(f"Predicted Digit: {prediction}")

    st.subheader("Prediction Probabilities")
    st.bar_chart(probs)

    st.subheader("Processed Input Image (28x28)")
    st.image(img_array, width=150, caption="Processed Image")

    st.subheader("First Conv Layer Feature Maps")
    cols = st.columns(8)
    num_features = min(16, feature_maps.shape[0])
    for i in range(num_features):
        col = cols[i % 8]
        fmap = feature_maps[i].cpu().numpy()
        fmap = (fmap - fmap.min()) / (fmap.max() - fmap.min() + 1e-5)
        col.image(fmap, width=64)


