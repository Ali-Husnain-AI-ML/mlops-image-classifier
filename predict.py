import tensorflow as tf
import numpy as np

# Load trained model
model = tf.keras.models.load_model("models/image_classifier.keras")

# Class names
class_names = [
    "buildings",
    "forest",
    "glacier",
    "mountain",
    "sea",
    "street"
]

# Image path
image_path = "data/mountain/19343.jpg"

# Load image
image = tf.keras.utils.load_img(
    image_path,
    target_size=(150, 150)
)

# Convert image to array
image_array = tf.keras.utils.img_to_array(image)

# Add batch dimension
image_array = np.expand_dims(image_array, axis=0)

# Make prediction
predictions = model.predict(image_array)

# Get predicted class
predicted_index = np.argmax(predictions[0])
predicted_class = class_names[predicted_index]

# Get confidence
confidence = predictions[0][predicted_index] * 100

print("\nPrediction:", predicted_class)
print("Confidence:", f"{confidence:.2f}%")