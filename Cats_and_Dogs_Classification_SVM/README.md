# 🐱🐶 Cats vs Dogs Image Classification using SVM

This project uses a **Support Vector Machine (SVM)** to classify images of **cats** and **dogs** from the popular Kaggle dataset [Dogs vs. Cats](https://www.kaggle.com/c/dogs-vs-cats). The images are preprocessed, flattened into feature vectors, and then used to train a simple linear classifier.

---

## 📌 Project Overview

- **Goal**: Classify whether an image is a cat or a dog.
- **Model**: Support Vector Machine (SVM)
- **Dataset**: [Kaggle Dogs vs Cats Dataset](https://www.kaggle.com/c/dogs-vs-cats/data)
- **Preprocessing**:
  - Resized images to 64x64
  - Converted to grayscale
  - Flattened image pixels into 1D arrays

---

## 🗃️ Dataset Preparation

1. Download the dataset from Kaggle:
   - [Kaggle Dogs vs Cats Dataset](https://www.kaggle.com/c/dogs-vs-cats/data)
   - Extract `train.zip`

2. Organize into folders:
dataset/
├── cats/
│ ├── cat.0.jpg
│ └── ...
└── dogs/
├── dog.0.jpg
└── ...

🧠 Model Details
Classifier: sklearn.svm.SVC

Kernel: 'linear'

Training/Testing Split: 80/20

Input: Flattened grayscale image of size 64x64 (4096 features)

📈 Sample Output
markdown
Copy
Edit
Accuracy: 87.2%

Classification Report:
              precision    recall  f1-score   support

        Cat       0.87      0.88      0.87       200
        Dog       0.87      0.86      0.86       200

    accuracy                           0.87       400


🚀 Future Improvements
Use HOG features for better feature extraction

Switch to a Convolutional Neural Network (CNN) for large datasets

Add GUI or live webcam testing

📚 Credits
Dataset: Kaggle Dogs vs Cats


