import streamlit as st
import pandas as pd
import numpy as np
import cv2
from PIL import Image
from prediction import predict_image
import joblib

st.title("Classifying Fruits")
st.markdown("Toy model to play to classify fruits into Apple, Banana, Cherry, Kiwi, Orange, Pineapple, Strawberries \
            base on image")
st.header("Plant Features")

# Cho phép người dùng tải lên một ảnh
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Đọc ảnh từ tệp tải lên
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Chuyển đổi ảnh sang định dạng OpenCV
    image = np.array(image)

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized_image = cv2.resize(gray_image, (256, 256))
    resized_image = resized_image.reshape((1, -1))  # Chuyển đổi thành dạng 2D (1, số lượng đặc trưng)
else:
    st.write("Please upload an image file")

if st.button("Predict type of Fruit") and uploaded_file is not None:
    # Load model và categories
    model = joblib.load("./voting_model.pkl")
    categories = ["Apple", "Banana", "Cherry", "Kiwi", "Orange", "Pineapple", "Strawberries"]

    # Dự đoán loại quả
    result = predict_image(model, categories, resized_image)
    st.text(f"The predicted label is: {result}")
