# Multi-Class Classification using CNN

This project demonstrates how to build and train a **Convolutional Neural Network (CNN)** for a **multi-class classification** task using TensorFlow and Keras. The model is trained on a dataset containing multiple classes and is evaluated on its ability to classify unseen data accurately.

## 📁 File Structure

- `multi-class_classification_CNN.ipynb` – Jupyter Notebook containing full code for data preprocessing, model building, training, and evaluation.

## 🚀 Features

- Data loading and preprocessing
- CNN model construction using Keras Sequential API
- Training with accuracy and loss visualization
- Model evaluation with classification report and confusion matrix

## 🛠️ Technologies Used

- Python
- TensorFlow / Keras
- NumPy
- Pandas
- Matplotlib / Seaborn
- Scikit-learn

## 🧠 Model Architecture

```python
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, CHANNELS)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(NUM_CLASSES, activation='softmax')
])
```
## 📊 Evaluation Metrics
- Accuracy

- Confusion Matrix

- Classification Report (Precision, Recall, F1-score)

## ⚙️ How to Run
1. Clone the repository or download the .ipynb file.
2. Make sure you have the required libraries installed:
  ```python
      pip install tensorflow matplotlib seaborn scikit-learn
  ```
3. Open the notebook and run all cells:
   ```python
         jupyter notebook multi-class_classification_CNN.ipynb
    ```
## 🧪 Sample Output
- Training & validation accuracy and loss plots
- Confusion matrix and classification report
- Final test accuracy

## 📌 Notes
- Ensure your dataset is preprocessed into proper shape (e.g., (num_samples, height, width, channels)).
- Update NUM_CLASSES, IMG_HEIGHT, IMG_WIDTH, and CHANNELS based on your dataset.
