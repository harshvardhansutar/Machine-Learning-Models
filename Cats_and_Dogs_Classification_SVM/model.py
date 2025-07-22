import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

# Constants
IMAGE_SIZE = (64, 64)  # Resize to 64x64 for faster training
DATASET_PATH = "dataset/"  # Path to your dataset folder

# Load and preprocess images
def load_images(folder):
    X, y = [], []
    for label, category in enumerate(['cats', 'dogs']):
        path = os.path.join(folder, category)
        for file in os.listdir(path):
            img_path = os.path.join(path, file)
            try:
                img = cv2.imread(img_path)
                img = cv2.resize(img, IMAGE_SIZE)
                img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                X.append(img_gray.flatten())  # flatten to 1D array
                y.append(label)
            except:
                continue
    return np.array(X), np.array(y)

print("[INFO] Loading images...")
X, y = load_images(DATASET_PATH)

# Split dataset
print("[INFO] Splitting dataset...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train SVM
print("[INFO] Training SVM...")
model = SVC(kernel='linear')  # You can try 'rbf' too
model.fit(X_train, y_train)

# Predict
print("[INFO] Evaluating model...")
y_pred = model.predict(X_test)

# Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=["Cat", "Dog"]))
