import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.models import load_model
import pandas as pd

# Load the saved model
model = load_model("leaf_counting_model_2.h5")  # Replace with the path to your saved model

# Function to generate Grad-CAM heatmap
def grad_cam(input_model, image, layer_name):
    grad_model = tf.keras.models.Model([input_model.inputs], [input_model.get_layer(layer_name).output, input_model.output])
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(np.array([image]))
        loss = predictions[:, np.argmax(predictions[0])]
    output = conv_outputs[0]
    grads = tape.gradient(loss, conv_outputs)[0]
    gate_f = tf.cast(output > 0, 'float32')
    gate_r = tf.cast(grads > 0, 'float32')
    guided_grads = gate_f * gate_r * grads
    weights = tf.reduce_mean(guided_grads, axis=(0, 1))
    cam = np.dot(output, weights)
    cam = cv2.resize(cam, (image.shape[1], image.shape[0]))  # Resize to match input image size
    cam = np.maximum(cam, 0)
    cam = cam / cam.max()
    return cam

# Function to predict leaf count from an image and generate Grad-CAM heatmap
def predict_leaf_count_with_explanation(image_path):
    # Load and preprocess the image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB (Matplotlib uses RGB)
    image_resized = cv2.resize(image, (128, 128))  # Resize to match model input size
    image_normalized = image_resized / 255.0  # Normalize pixel values

    # Use the model to predict leaf count
    predicted_leaf_count = model.predict(np.expand_dims(image_normalized, axis=0))[0][0]

    # Generate Grad-CAM heatmap
    cam = grad_cam(model, image_normalized, 'conv2d')

    # Overlay heatmap on the original image
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(image_resized, 0.5, heatmap, 0.5, 0)

    return predicted_leaf_count, heatmap, superimposed_img

# Function to read actual leaf counts from Excel file
def get_actual_leaf_count_from_excel(image_filename, excel_data):
    row = excel_data.loc[excel_data['fn'] == image_filename]
    return row['ln'].values[0] if not row.empty else None

# Directory containing images
images_folder = "images_test"

# Load leaf count data from Excel file
excel_file = "sorghum_leaf_number.xlsx"  # Adjust the file path as necessary
leaf_counts_data = pd.read_excel(excel_file)

# Predict leaf count and generate Grad-CAM explanation for each image in the folder
for filename in os.listdir(images_folder):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        image_path = os.path.join(images_folder, filename)
        predicted_count, heatmap, explanation_image = predict_leaf_count_with_explanation(image_path)

        # Get actual leaf count from Excel file
        actual_leaf_count = get_actual_leaf_count_from_excel(filename, leaf_counts_data)

        # Display the original image, heatmap, and superimposed image
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 3, 1)
        plt.imshow(cv2.imread(image_path))
        plt.title('Original Image\nActual Count: {}'.format(actual_leaf_count))
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.imshow(heatmap, cmap='jet')
        plt.title('Heatmap')
        plt.colorbar(label='Intensity')  # Add color bar with label
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        plt.imshow(explanation_image)
        plt.title('Superimposed Image\nPredicted Count: {:.2f}'.format(predicted_count))
        plt.axis('off')
        
        plt.show()
