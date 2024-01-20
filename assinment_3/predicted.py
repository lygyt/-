from keras.models import load_model
import numpy as np
import cv2
import os

# Function to preprocess the input image
def preprocess_image(image_path):
    # Read the image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Resize the image to 28x28 pixels
    img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)

    # Invert the image (black digits on white background)
    img = 255 - img

    # Normalize the image
    img = img / 255.0

    # Flatten the image and return
    return img.reshape(-1, 28 * 28)

# Load the saved model
model = load_model('assinment_3\keras_minist_dense.model',compile=False)
model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])
# Path to the folder containing input imag  es
input_folder_path = r'assinment_3\split'

# Iterate through images in the folder
for file in os.listdir(input_folder_path):
    if file.endswith(".png"):  # Assuming images are in PNG format
        # Construct the path to each image
        input_image_path = os.path.join(input_folder_path, file)

        # Preprocess the input image
        input_data = preprocess_image(input_image_path)

        # Make a prediction
        prediction = model.predict(input_data)
        predicted_class = np.argmax(prediction, axis=1)

        print(f"Prediction for {input_image_path}: {predicted_class[0]}")

