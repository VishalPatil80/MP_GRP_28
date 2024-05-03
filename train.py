import pandas as pd
import numpy as np 
import os
from keras.preprocessing.image import load_img, img_to_array
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error
from keras import regularizers
from keras.callbacks import ModelCheckpoint

# Load the annotations from the Excel file
excel_file = "sorghum_leaf_number.xlsx"
annotations = pd.read_excel(excel_file)

# Filter out annotations for images that you have available
available_images_folder = "sorghum_10_number"
available_images = os.listdir(available_images_folder)
annotations = annotations[annotations['fn'].isin(available_images)]

# Function to preprocess images
def preprocess_image(img_path, target_size=(128, 128)):
    img = load_img(img_path, target_size=target_size)
    img_array = img_to_array(img)
    return img_array / 255.0  # Normalize pixel values to [0, 1]

# Preprocess images and extract labels
images = []
labels = []
for index, row in annotations.iterrows():
    img_path = os.path.join(available_images_folder, row['fn'])
    images.append(preprocess_image(img_path))
    labels.append(row['ln'])  # Assuming 'angle' column contains the count of leaves

images = np.array(images)
labels = np.array(labels)

# Split the dataset into training, validation, and testing sets
X_train, X_temp, y_train, y_temp = train_test_split(images, labels, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Build the CNN model with increased capacity and regularization
model = Sequential([
    Conv2D(256, (3, 3), activation='relu', input_shape=(128, 128, 3)), 
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'), 
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'), 
    MaxPooling2D((2, 2)),
    Conv2D(32, (3, 3), activation='relu'), 
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu', kernel_regularizer=regularizers.l1(0.01)), 
    Dense(64, activation='relu', kernel_regularizer=regularizers.l1(0.01)),
    Dense(32, activation='relu', kernel_regularizer=regularizers.l1(0.01)), 
    Dense(1)
])

# Compile the model with a lower learning rate
opt = tf.keras.optimizers.Adam(lr=0.0001)
model.compile(optimizer=opt, loss='mean_squared_error', metrics=['mae'])

# Define a ModelCheckpoint callback to save the model after the 3rd epoch
checkpoint = ModelCheckpoint("leaf_counting_model_epoch_3_final.h5", monitor='val_loss', verbose=1, save_best_only=False, save_weights_only=False, mode='auto', period=3)

# Train the model with increased epochs
history = model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_val, y_val), callbacks=[checkpoint])

# Evaluate the model
loss, mae = model.evaluate(X_test, y_test)
print("Test Mean Absolute Error:", mae)

# Scatter Plot of Predicted vs. Actual Values
predicted_values = model.predict(X_test).flatten()
plt.figure(figsize=(8, 8))
plt.scatter(y_test, predicted_values, alpha=0.5)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Scatter Plot of Predicted vs. Actual Values')
plt.show()

# Mean Absolute Error (MAE) Distribution
mae_values = np.abs(predicted_values - y_test)
plt.figure(figsize=(8, 6))
plt.hist(mae_values, bins=20, edgecolor='black')
plt.xlabel('Mean Absolute Error (MAE)')
plt.ylabel('Frequency')
plt.title('Mean Absolute Error (MAE) Distribution')
plt.show()

# Root Mean Squared Error (RMSE)
mse = mean_squared_error(y_test, predicted_values)
rmse = np.sqrt(mse)
print("Test Root Mean Squared Error:", rmse)

# Learning Curve
train_loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(train_loss) + 1)
plt.figure(figsize=(10, 6))
plt.plot(epochs, train_loss, 'b', label='Training Loss')
plt.plot(epochs, val_loss, 'r', label='Validation Loss')
plt.title('Learning Curve')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Residual Plot
residuals = predicted_values - y_test
plt.figure(figsize=(8, 6))
plt.scatter(predicted_values, residuals, alpha=0.5)
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.axhline(y=0, color='r', linestyle='--')
plt.show()

# Save the trained model
model.save("leaf_counting_model_final.h5")
