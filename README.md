# 🧠 Brain Tumor Detection Using CNN

A deep learning-based project that uses Convolutional Neural Networks (CNN) to classify MRI brain images into four categories:

- **Glioma**
- **Meningioma**
- **Pituitary**
- **No Tumor**

This project applies **image preprocessing**, **data augmentation**, and **hyperparameter tuning** techniques to enhance model performance and efficiently predict the presence and type of brain tumors.

---

## 📂 Project Folder Structure

```
brain-tumor-detection/
├── dataset/                    # Dataset folder (Training & Testing images)
├── images/                     # Saved model performance visualizations (accuracy/loss plots)
├── models/                     # Trained models
│   └── brain_tumor_cnn.keras
├── notebooks/                  # Jupyter Notebook version of the project
│   └── brain_tumor_detection.ipynb
├── main.py                     # Python script version for training & evaluation
├── README.md                   # Project overview and instructions
└── requirements.txt            # Required Python libraries
```

---

## 📊 Dataset Information

- **Dataset Source:**  
  [Brain MRI Dataset (Kaggle)](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)

- **Dataset Structure:**
  - **Training Folder:**
    - Glioma
    - Meningioma
    - Pituitary
    - No Tumor
  - **Testing Folder:**
    - Same structure as Training folder.

---

## 🚀 Technologies Used

- **Programming Language:** Python
- **Deep Learning Framework:** TensorFlow & Keras
- **Libraries:** NumPy, Pandas, Matplotlib, OpenCV
- **Model Type:** Convolutional Neural Network (CNN)
- **Techniques:**
  - Data Augmentation
  - Image Rescaling
  - Hyperparameter Tuning
  - Visualization of Accuracy & Loss

---

## 🛠️ How to Set Up & Run

### 🔽 1. Clone the Repository

```bash
git clone https://github.com/krishnagiri1/brain-tumor-detection.git
cd brain-tumor-detection
```

---

### 📦 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 🏃 3. Run Project

#### Option 1: Run Python Script

```bash
python main.py
```

#### Option 2: Open Jupyter Notebook

```bash
cd notebooks
jupyter notebook brain_tumor_detection.ipynb
```

---

## 📈 Results

| Metric                | Result        |
|----------------------|--------------|
| **Final Test Accuracy**| **97%**       |
| Model Used           | CNN (3 Conv layers + Dense layers) |
| Input Image Size     | 128x128       |
| Optimizer            | Adam          |
| Loss Function        | Categorical Crossentropy |

---

## 📌 Future Improvements

- Implement **Transfer Learning** using pre-trained models (VGG, ResNet).
- Further fine-tuning with advanced optimizers.
- Deploy the model using Flask/Django web application.
- Apply cross-validation for robust evaluation.

---

## 🙌 Acknowledgments

- Dataset credits: Masoud Nickparvar (Kaggle)
- TensorFlow, Keras documentation for CNN architecture inspiration.

---

## 📃 License

This project is licensed for educational purposes only.

---

## ⭐ If you like this project, feel free to give it a star ⭐ and contribute!
