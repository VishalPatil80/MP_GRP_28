import numpy as np
import os
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf

# Load the saved model
model = load_model("leaf_counting_model_2.h5")  # Replace "leaf_counting_model.h5" with the path to your saved model

# Function to preprocess images
def preprocess_image(img_path, target_size=(128, 128)):
    img = load_img(img_path, target_size=target_size)
    img_array = img_to_array(img)
    return img_array / 255.0  # Normalize pixel values to [0, 1]

# Function to predict leaf count from an image and generate Grad-CAM heatmap
def predict_leaf_count_with_explanation(image_path):
    # Preprocess the image
    image = preprocess_image(image_path)

    # Reshape the image to match the input shape expected by the model
    image = np.expand_dims(image, axis=0)

    # Use the model to predict leaf count
    predicted_leaf_count = model.predict(image)[0][0]

    # Generate Grad-CAM heatmap
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
        cam = cv2.resize(cam, (128, 128))
        cam = np.maximum(cam, 0)
        cam = cam / cam.max()
        return cam

    # Generate Grad-CAM heatmap for the first convolutional layer
    cam = grad_cam(model, image[0], 'conv2d')

    # Overlay heatmap on the original image
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted((image[0] * 255).astype('uint8'), 0.5, heatmap, 0.5, 0)

    return predicted_leaf_count, superimposed_img

# Example filename
filename = "predict.jpg"  # Replace with the actual filename of the image you want to predict the leaf count for

# Path to the image
image_path = "predict.jpg"  # Assuming available_images_folder is defined

# Predict leaf count and generate Grad-CAM explanation
predicted_count, explanation_image = predict_leaf_count_with_explanation(image_path)
print("Predicted Leaf Count:", predicted_count)

# Display the original image and Grad-CAM explanation
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(load_img(image_path))
plt.title('Original Image')
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(explanation_image, cv2.COLOR_BGR2RGB))
plt.title('Grad-CAM Explanation')
plt.axis('off')
plt.show()