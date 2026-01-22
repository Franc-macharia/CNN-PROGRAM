import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import numpy as np



# ----------------------------------------------------------
# PART 1: TRAIN THE MODEL
# ----------------------------------------------------------
def train_model():
    # Data generators
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )
    test_datagen = ImageDataGenerator(rescale=1./255)

    # Load dataset from folder
    train_data = train_datagen.flow_from_directory(
        "cat_dog_dataset/train",
        target_size=(150, 150),
        batch_size=32,
        class_mode="binary"
    )

    test_data = test_datagen.flow_from_directory(
        "cat_dog_dataset/test",
        target_size=(150, 150),
        batch_size=32,
        class_mode="binary"
    )

    # Build CNN model
    model = Sequential([
        Conv2D(32, (3,3), activation="relu", input_shape=(150,150,3)),
        MaxPooling2D(2,2),

        Conv2D(64, (3,3), activation="relu"),
        MaxPooling2D(2,2),

        Conv2D(128, (3,3), activation="relu"),
        MaxPooling2D(2,2),

        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.5),
        Dense(1, activation="sigmoid")
    ])

    # Compile model
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    # Train model
    model.fit(train_data, epochs=10, validation_data=test_data)

    # Save model
    model.save("cat_dog_model.h5")
    print("✅ Model training completed and saved as cat_dog_model.h5")


# ----------------------------------------------------------
# PART 2: PREDICT CAT OR DOG
# ----------------------------------------------------------
def predict_image(img_path):
    # Load trained model
    model = load_model("cat_dog_model.h5")

    # Load and preprocess image
    img = load_img(img_path, target_size=(150,150))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)[0]
    if prediction > 0.5:
        print("Prediction: DOG 🐶")
    else:
        print("Prediction: CAT 🐱")


# ----------------------------------------------------------
# MAIN MENU
# ----------------------------------------------------------
if __name__ == "__main__":
    print("Choose an option:")
    print("1. Train model")
    print("2. Predict an image")

    choice = input("Enter 1 or 2: ")

    if choice == "1":
        train_model()
    elif choice == "2":
        img_path = input("Enter image path: ")
        predict_image(img_path)
    else:
        print("Invalid choice")
