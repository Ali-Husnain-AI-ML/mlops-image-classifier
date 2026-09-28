import tensorflow as tf

# ==========================================
# 1. Dataset Path
# ==========================================

DATASET_PATH = "data"

print("Loading dataset...")


# ==========================================
# 2. Load Training Dataset
# ==========================================

train_dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=(150, 150),
    batch_size=32
)


# ==========================================
# 3. Load Validation Dataset
# ==========================================

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=(150, 150),
    batch_size=32
)

print("Dataset loaded successfully!")
print("Classes:", train_dataset.class_names)
print("Training batches:", len(train_dataset))
print("Validation batches:", len(validation_dataset))


# ==========================================
# 4. Build CNN Model
# ==========================================

model = tf.keras.Sequential([

    # Normalize pixel values
    tf.keras.layers.Rescaling(
        1./255,
        input_shape=(150, 150, 3)
    ),

    # First Convolution Block
    tf.keras.layers.Conv2D(
        32,
        (3, 3),
        activation="relu"
    ),
    tf.keras.layers.MaxPooling2D(),

    # Second Convolution Block
    tf.keras.layers.Conv2D(
        64,
        (3, 3),
        activation="relu"
    ),
    tf.keras.layers.MaxPooling2D(),

    # Third Convolution Block
    tf.keras.layers.Conv2D(
        128,
        (3, 3),
        activation="relu"
    ),
    tf.keras.layers.MaxPooling2D(),

    # Convert feature maps into one vector
    tf.keras.layers.Flatten(),

    # Fully connected layer
    tf.keras.layers.Dense(
        128,
        activation="relu"
    ),

    # Reduce overfitting
    tf.keras.layers.Dropout(0.5),

    # Output layer
    # 6 classes
    tf.keras.layers.Dense(
        6,
        activation="softmax"
    )
])


# ==========================================
# 5. Compile Model
# ==========================================

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("\nCNN model created successfully!")


# ==========================================
# 6. Display Model Information
# ==========================================

model.summary()


# ==========================================
# 7. Train Model
# ==========================================

print("\nStarting model training...\n")

history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=10
)


# ==========================================
# 8. Evaluate Model
# ==========================================

print("\nEvaluating model...")

loss, accuracy = model.evaluate(validation_dataset)

print("\nValidation Loss:", loss)
print("Validation Accuracy:", accuracy)


# ==========================================
# 9. Save Model
# ==========================================

model.save("models/image_classifier.keras")

print("\nModel saved successfully!")
print("Location: models/image_classifier.keras")