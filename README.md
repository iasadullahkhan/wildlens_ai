# 🦁 WildLens AI — Wildlife Species Recognition System

## 📌 Overview

**WildLens AI** is an end-to-end deep learning-based wildlife species classification system designed as a Final Year Project (FYP). The system uses a Convolutional Neural Network (CNN) model to identify wildlife species from images and provides explainable AI outputs using Grad-CAM visualizations.

The project combines **Machine Learning, Computer Vision, and Full-Stack Web Development (Streamlit)** into a complete intelligent wildlife monitoring system.

It supports:

* Single and batch image prediction
* 12 wildlife species classification
* AI explainability (Grad-CAM heatmaps)
* Analytics dashboard
* Prediction history tracking
* Data export functionality

---

## 🚀 Features

### 🔍 AI-Based Wildlife Classification

* Predicts 12 different wildlife species
* Deep Learning CNN model (`model.h5`)
* High accuracy image classification

### 🔥 Explainable AI (Grad-CAM)

* Visual heatmaps showing model attention areas
* Helps interpret AI decisions

### 📊 Analytics Dashboard

* Species distribution charts
* Confidence score analysis
* Class-wise statistics (Mammals vs Birds)
* Performance insights

### 📸 Batch Image Processing

* Upload multiple images at once
* Real-time processing with progress tracking

### 📜 Prediction History System

* Stores all predictions during session
* Tracks confidence, species, and timestamps
* Search and filter functionality

### 📥 Export System

* Export predictions to CSV format
* Download species and analytics reports

### 🎨 Modern UI

* Fully customized Streamlit interface
* Dark-themed professional dashboard
* Responsive card-based design

---

## 🧠 Tech Stack

### 🔹 Frontend / UI

* Streamlit
* HTML + CSS (custom styling)
* Streamlit Components

### 🔹 Backend / ML

* TensorFlow / Keras
* OpenCV
* NumPy

### 🔹 Data Processing

* Pandas
* Matplotlib
* Seaborn

### 🔹 Explainability

* Grad-CAM (Custom implementation)

### 🔹 Other Tools

* PIL (Image processing)
* Base64 encoding
* LocalStorage integration (browser sync)

---

## 🧬 System Architecture

```
Input Image
     ↓
Preprocessing (Resize → Normalize)
     ↓
CNN Model (model.h5)
     ↓
Prediction (12 Classes)
     ↓
Grad-CAM Heatmap Generation
     ↓
Visualization + Analytics Dashboard
     ↓
History Storage (Session + LocalStorage)
```

---

## 📂 Project Structure

```
WildLens-AI/
│
├── app.py                  # Main Streamlit application
├── model.h5               # Trained CNN model
├── class_names.txt        # List of class labels
├── model_utils.py         # Prediction helper functions
├── gradcam.py             # Grad-CAM heatmap generation
│
├── assets/                # Images / UI assets (optional)
├── data/                  # Dataset (if included)
│
└── README.md              # Project documentation
```

---

## 🧪 Model Details

* **Architecture:** CNN (Convolutional Neural Network)
* **Input Size:** 224 × 224 RGB images
* **Output:** Softmax (12 classes)
* **Loss Function:** Categorical Crossentropy
* **Optimizer:** Adam

### 📌 Species Classes:

* Common Leopard
* Snow Leopard
* Grey Langur
* Jackal
* Jungle Cat
* Leopard Cat
* Mongoose
* Porcupine
* Red Fox
* Rhesus Macaque
* Wild Boar
* Kalij Pheasant

---

## 📊 Analytics Features

The system provides detailed insights:

* Total predictions made
* Average confidence score
* Species frequency distribution
* Class distribution (Mammals vs Birds)
* Confidence level breakdown (High / Medium / Low)
* Top high-confidence predictions

---

## 🖥️ How to Run the Project

### 1️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 2️⃣ Run the Application

```bash
streamlit run app.py
```

---

### 3️⃣ Open in Browser

```
http://localhost:8501
```

---

## 📷 How It Works

1. Upload wildlife image(s)
2. Image is preprocessed (224×224)
3. CNN model predicts species
4. Grad-CAM generates heatmap
5. Results are displayed visually
6. Data is stored in session history
7. Analytics dashboard updates in real-time

---

## 📈 Key Highlights (FYP Strength)

✔ End-to-end AI system
✔ Real-time prediction interface
✔ Explainable AI (Grad-CAM)
✔ Interactive analytics dashboard
✔ Batch processing capability
✔ Data export functionality
✔ Professional UI/UX design

---

## ⚠️ Limitations

* Model depends on dataset quality
* No cloud deployment (local execution)
* Session-based history (not database-backed)
* Requires GPU for faster training/inference (optional)

---

## 🔮 Future Improvements

* Cloud deployment (AWS / Azure / Streamlit Cloud)
* Database integration (PostgreSQL / Firebase)
* Mobile application version
* Real-time camera detection system
* Larger dataset expansion
* Model optimization (EfficientNet / Vision Transformers)

---

## 👨‍💻 Developer

**Asad Ullah Khan**
BS Information Technology (Final Year Project)

---

## 📜 License

This project is created for academic purposes as a Final Year Project.
All rights reserved © 2026.

---

## ⭐ Acknowledgements

* TensorFlow & Keras community
* Streamlit framework
* OpenCV library
* Research papers on CNN & Grad-CAM

---
