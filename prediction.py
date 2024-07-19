import cv2
import numpy as np

def predict_image(model, categories, image_data):
    # Tiền xử lý hình ảnh từ dữ liệu
    if image_data is None:
        raise ValueError("No image data provided")

    # Dự đoán nhãn
    y_pred = model.predict(image_data)

    # Lấy tên nhãn từ số nhãn dự đoán
    label = categories[y_pred[0]]

    return label
