from keras.models import load_model
import numpy as np
import os

import matplotlib.pyplot as plt

# Function to read and preprocess the input image
def read_image(add, nums):
    x_train = np.zeros(nums * 28 * 28)
    with open(add, 'rb') as image:
        image.seek(16)
        data = image.read(nums * 28 * 28)
        for i in range(nums * 28 * 28):
            # x_train[i] = data[i] / 255
            x_train[i] = data[i] / 255    
    # x_train 处理结束
    return x_train.reshape(-1, 28 * 28)

# Path to the folder containing input images
input_folder_path = r'assinment_3\split'

# Load the saved model
model = load_model(r'assinment_3\keras_minist_dense.model')

# Iterate through images in the folder
for i in range(10):  # Assuming there are 10 images named 0.png, 1.png, ..., 9.png
    # Construct the path to each image
    input_image_path = os.path.join(input_folder_path, f"{i}.png")

    # Read and preprocess the input image
    input_data = read_image(input_image_path, 1)

    # Make a prediction
    classes = np.argmax(model.predict(input_data))

    print(f"Prediction for {input_image_path}: {classes-1}")
