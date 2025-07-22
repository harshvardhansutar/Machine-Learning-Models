import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Set dataset path
dataset_path = r"P:\Machine_Learning_Models\Hand_Gesture_Recognition\dataset\leapGestRecog"
image_size = 64

# Initialize data
X = []
y = []

print("[INFO] Loading dataset...")

# Create a consistent label mapping dictionary
label_map = {}   # e.g., {"01_palm": 0, "02_l": 1, ..., "10_fist_moved": 9}
label_counter = 0

# Loop through all subject folders (e.g., 00, 01, ..., 09)
for subject in os.listdir(dataset_path):
    subject_path = os.path.join(dataset_path, subject)
    if not os.path.isdir(subject_path):
        continue
    
    # Loop through gesture class folders (e.g., 01_palm, etc.)
    for gesture in os.listdir(subject_path):
        gesture_path = os.path.join(subject_path, gesture)
        if not os.path.isdir(gesture_path):
            continue

        # Assign numeric label
        if gesture not in label_map:
            label_map[gesture] = label_counter
            label_counter += 1
        
        numeric_label = label_map[gesture]

        for image_name in os.listdir(gesture_path):
            if image_name.endswith(".png"):
                image_path = os.path.join(gesture_path, image_name)
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                image = cv2.resize(image, (image_size, image_size))
                X.append(image)
                y.append(numeric_label)

# Check if data loaded
if len(X) == 0 or len(y) == 0:
    raise ValueError("No data loaded. Check your dataset path and folder structure.")

# Convert to numpy arrays and normalize
X = np.array(X).reshape(-1, image_size, image_size, 1) / 255.0
y = to_categorical(y, num_classes=10)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(image_size, image_size, 1)),
    MaxPooling2D((2, 2)),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),

    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print("[INFO] Training model...")
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Save model
model.save("hand_gesture_model.h5")
print("[INFO] Model saved as hand_gesture_model.h5")

# Optional: Print label mapping
print("[INFO] Label Mapping:")
for gesture, idx in label_map.items():
    print(f"{idx} -> {gesture}")
