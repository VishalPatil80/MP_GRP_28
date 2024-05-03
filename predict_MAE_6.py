import os
import cv2
import numpy as np
import tensorflow as tf
from keras.models import load_model
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Load the saved model
model = load_model("leaf_counting_model_2.h5")  # Replace with the path to your saved model

# Function to preprocess the image
def preprocess_image(image):
    image_resized = cv2.resize(image, (128, 128))  # Resize to match model input size
    image_normalized = image_resized / 255.0  # Normalize pixel values
    return image_normalized

# Function to predict leaf count from an image
def predict_leaf_count(image):
    image_normalized = preprocess_image(image)
    predicted_leaf_count = model.predict(np.expand_dims(image_normalized, axis=0))[0][0]
    return predicted_leaf_count

# Function to read actual leaf counts from Excel file
def get_actual_leaf_count_from_excel(image_filename, excel_data):
    row = excel_data.loc[excel_data['fn'] == image_filename]
    return row['ln'].values[0] if not row.empty else None

# Directory containing images
images_folder = "sorghum_10_number"

# Load leaf count data from Excel file
excel_file = "sorghum_leaf_number.xlsx"  # Adjust the file path as necessary
leaf_counts_data = pd.read_excel(excel_file)

predicted_counts = []
actual_counts = []

# Predict leaf count for each image in the folder
for filename in os.listdir(images_folder):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        image_path = os.path.join(images_folder, filename)
        image = cv2.imread(image_path)
        predicted_count = predict_leaf_count(image)
        actual_count = get_actual_leaf_count_from_excel(filename, leaf_counts_data)
        if actual_count is not None:
            predicted_counts.append(predicted_count)
            actual_counts.append(actual_count)

# Calculate RMSE and MAE
rmse = np.sqrt(mean_squared_error(actual_counts, predicted_counts))
mae = mean_absolute_error(actual_counts, predicted_counts)

print("Root Mean Squared Error (RMSE):", rmse)
print("Mean Absolute Error (MAE):", mae)
