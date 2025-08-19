# 🧠 MNIST Digit Recognition using Support Vector Machine (SVM)

This project implements a **Support Vector Machine (SVM)** classifier to recognize handwritten digits using the **MNIST dataset**.  
The model achieves **~97.9% accuracy** on the MNIST test set.  

It also includes a **custom prediction script (UI)** that allows you to provide your own handwritten digit image, preprocess it, and predict the digit.

---

## 📌 Features
- Preprocessing:
  - Checks and reports **null values** and **duplicate rows** in train/test datasets.
  - Renames pixel columns as `p1` … `p784` for clarity.
  - Normalizes pixel values with **MinMaxScaler**.
- Trains an **SVM classifier** on the MNIST dataset (60,000 train, 10,000 test).
- Evaluation with:
  - **Accuracy**
  - **Precision**
  - **Recall**
  - **F1 Score**
  - **Confusion Matrix** (visualized with a Seaborn heatmap).
- Saves and reloads **trained model** and **scaler** using Joblib.
- Custom **UI script (`predict.py`)**:
  - Loads any handwritten digit image.
  - Converts to grayscale and resizes to 28×28.
  - Inverts colors if needed (based on image mean).
  - Extracts bounding box, centers digit in a 28×28 canvas.
  - Predicts the digit using the trained SVM.
  - Displays the processed digit with Matplotlib.

---

## 📂 Project Structure
```
MNIST-SVM/
├── data/
│   ├── mnist_train.csv
│   ├── mnist_test.csv
├── models/
│   ├── model.pkl
│   ├── scaler.pkl
├── main.ipynb          # Training, evaluation, saving model & scaler
├── UI.ipynb        # UI for predicting custom handwritten digits
├── requirements.txt  # Dependencies
└── README.md         # Documentation
```

---

## 🚀 Getting Started

### 1️⃣ Clone the repository
```bash
git clone https://github.com/<your-username>/MNIST-SVM.git
cd MNIST-SVM
```

### 2️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Train the model
Run the training script (this will train SVM, evaluate, and save `model.pkl` + `scaler.pkl`):
```bash
python train.py
```

### 4️⃣ Predict a custom digit
Run the prediction script:
```bash
python predict.py
```
When prompted, paste the full path to your digit image.

---

## 📊 Model Performance
- **Accuracy:** ~97.9%
- **Precision:** ~97–98%
- **Recall:** ~97–98%
- **F1 Score:** ~97–98%

---

## 🔧 Notes on Implementation (important)
- The training script:
  - Loads `mnist_train.csv` and `mnist_test.csv`.
  - Checks for nulls and duplicates and prints those counts.
  - Renames columns to `p1` … `p784`.
  - Scales features using `MinMaxScaler`.
  - Trains `SVC()` (default kernel RBF). If you plan to use `predict_proba`, re-train with `probability=True`.
  - Evaluates using accuracy, precision, recall, f1, and displays a confusion matrix heatmap (Seaborn).
  - Saves model and scaler with `joblib.dump`.

- The prediction script:
  - Loads an input image, converts to grayscale and resizes to 28×28.
  - Checks mean to decide whether to invert colors.
  - Thresholding is **not** recommended (this project keeps grayscale information).
  - Extracts bounding box of the digit using OpenCV, applies Gaussian smoothing (optional) and resizes to 20×20, then centers it in a 28×28 canvas.
  - Flattens the canvas to (1,784), wraps into a DataFrame with columns `p1`…`p784`, and scales with the saved `MinMaxScaler` before predicting.

---

## 🔮 Next Steps
- **Future work** → Optionally implement a CNN for improved performance on custom handwritten digits.

---

## 📜 Requirements
The project requires the following Python libraries:
```
numpy
pandas
scikit-learn
seaborn
matplotlib
pillow
opencv-python
joblib
```

---

## 👨‍💻 Author
**Saifullah Umar**  
- 🎓 BS Artificial Intelligence student at Nutech University Islamabad, Pakistan 
- 💻 Machine Learning Engineer

---
