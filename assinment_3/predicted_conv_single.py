from keras.models import load_model
import numpy as np
import cv2
import os

def read_and_preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Image at {image_path} not read properly.")

    img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
    # Uncomment the following line if your model expects white digit on black background
    # img = 255 - img
    img = img / 255.0
    return img.reshape(-1, 28, 28, 1)

# Load the saved model
model = load_model(r"assinment_3\keras_minist.model")

# Path to the folder containing input images
input_folder_path = r'assinment_3\split'

# Iterate through images in the folder
for file_name in os.listdir(input_folder_path):
    if file_name.endswith((".png", ".jpg", ".jpeg")):  # Check if the file is an image
        image_path = os.path.join(input_folder_path, file_name)
        try:
            x_pred = read_and_preprocess_image(image_path)
            predictions = model.predict(x_pred)
            predicted_label = np.argmax(predictions, axis=1)
            print(f"Prediction for {file_name}: {predicted_label[0]}")
        except Exception as e:
            print(f"Error processing {file_name}: {e}")
