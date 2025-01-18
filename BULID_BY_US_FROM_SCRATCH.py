import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
import numpy as np
from tensorflow.keras.preprocessing import image
import cv2
import os

def train_or_load_model(train_dir):
    """Train new model or load existing one"""
    if os.path.exists("garbage_classification_model_enhanced.h5"):
        print("Loading existing model...")
        return load_model("garbage_classification_model_enhanced.h5")
    
    print("Training new model...")
    datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True
    )

    train_generator = datagen.flow_from_directory(
        train_dir,
        target_size=(150, 150),
        batch_size=32,
        class_mode='categorical',
        subset='training'
    )

    num_classes = len(train_generator.class_indices)

    model = Sequential([
        Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(150, 150, 3)),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        
        Conv2D(256, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam',
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])

    model.fit(train_generator,
              epochs=1,
              verbose=1)
    
    model.save("garbage_classification_model_enhanced.h5")
    return model

def classify_image(model, image_array, train_dir):
    """Classify a single image"""
    # Get class labels
    datagen = ImageDataGenerator(rescale=1./255)
    temp_generator = datagen.flow_from_directory(
        train_dir,
        target_size=(150, 150),
        batch_size=1
    )
    class_labels = list(temp_generator.class_indices.keys())
    
    # Preprocess and predict
    img_array = cv2.resize(image_array, (150, 150))
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    prediction = model.predict(img_array)
    class_idx = np.argmax(prediction)
    confidence = prediction[0][class_idx] * 100
    
    return class_labels[class_idx], confidence

def main():
    # Set up the model
    train_dir = r"C:\Users\anubh\Downloads\archive (2)\Garbage classification\Garbage classification"
    model = train_or_load_model(train_dir)
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    print("\nCamera feed is live!")
    print("Press 'c' to capture and classify")
    print("Press 'q' to quit")

    while True:
        # Read frame
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Show frame
        cv2.imshow('Camera Feed (Press c to capture, q to quit)', frame)

        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            # Save and classify the frame
            print("\nClassifying image...")
            class_name, confidence = classify_image(model, frame, train_dir)
            print(f"Predicted class: {class_name}")
            print(f"Confidence: {confidence:.2f}%")
            
            # Save the captured image
            cv2.imwrite('captured_image.jpg', frame)
            print("Image saved as 'captured_image.jpg'")
            
        elif key == ord('q'):
            print("Exiting...")
            break

    # Clean up
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
