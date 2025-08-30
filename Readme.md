# Leaf-Insight-Deep-Learning-for-Plant-Health

A Deep Learning–based **Leaf Disease Classifier** built with **TensorFlow/Keras** and deployed using **Streamlit**.  
This project classifies leaf images into **13 different disease categories or healthy leaf states**. Users can upload a leaf image or capture a live image through their webcam for real-time plant health diagnosis.

---

## 📌 Features
- ✅ Classifies **13 leaf disease and healthy leaf classes**  
- ✅ Supports both **image upload** and **live camera capture**  
- ✅ Built using an effective **Convolutional Neural Network (CNN)** architecture  
- ✅ Utilizes **data augmentation** to improve model generalization  
- ✅ Simple, user-friendly **Streamlit Web Application** interface  
- ✅ Easy to deploy and run locally  

---

## ⚡ Usage Overview
1. Launch the Streamlit app  
2. Select **Upload Image** or **Capture Live Image**  
3. The model predicts the leaf's disease class or healthy status and displays the result

---

## 📂 Repository Structure

```
Leaf-Insight-Deep-Learning-for-Plant-Health/
│── app.py # Streamlit app for inference
│── leaf_disease_classifier.h5 # Pretrained CNN model file
│── README.md # Project documentation
│── dataset/ # (Optional) Leaf disease dataset folder
```


---

## 📦 Installation & Setup

### 1️⃣ Clone the Repository

```
git clone https://github.com/omjaikumar23/Leaf-Insight-Deep-Learning-for-Plant-Health.git
cd Leaf-Insight-Deep-Learning-for-Plant-Health
```


### 2️⃣ Create a Virtual Environment (Recommended)

**Windows (PowerShell):**

```
python -m venv .venv
.venv\Scripts\activate
```

**Mac/Linux:**

```
python3 -m venv .venv
source .venv/bin/activate
```


### 3️⃣ Install Dependencies

```
pip install tensorflow streamlit pillow numpy
```

### 4️⃣ Run the Streamlit App

```
streamlit run app.py
```


### 5️⃣ Access the App  
Once running, Streamlit will provide a local URL where you can interact with the app.

---

## 🗂 Dataset Details & Usage

### Source  
This project uses the **Leaf Disease Collection dataset** containing labeled images of leaves categorized by various diseases and healthy states, split into training, validation, and test sets.

### Dataset Preparation  
- Images resized to 256x256 pixels  
- Data augmentation applied including rotations, shifts, and flips for better generalization  
- Organized into directories for training, validation, and testing compatible with TensorFlow's `ImageDataGenerator`

---

## 🏋️ Model Training Details
- CNN architecture with layers of Conv2D, MaxPooling, GlobalAveragePooling, Dense  
- Input: 256x256 RGB images  
- Optimizer: **RMSprop** with learning rate 0.001  
- Loss function: **Categorical Crossentropy**  
- Metrics: **Accuracy**  
- Training epochs: 10 (can be increased for improved accuracy)  
- Model saved as `leaf_disease_classifier.h5`

---

## 🎮 How to Use
- Upload an image file (`.jpg`, `.jpeg`, `.png`) to get a prediction of the leaf disease or healthy class  
- Alternatively, capture a live image using your webcam for instant prediction  

---

## 🔧 Tech Stack
- Python 3.x  
- TensorFlow / Keras  
- Streamlit  
- PIL (Pillow)  
- NumPy  

---

## 🚀 Future Improvements
- Deploy on platforms such as **Streamlit Cloud** or **Hugging Face Spaces**  
- Apply **Transfer Learning** (e.g., MobileNet, ResNet50) for enhanced accuracy  
- Add **explainability tools** like Grad-CAM for better model interpretability  
- Extend classification to additional plant diseases and species  

---

## 🤝 Contributing
Contributions are welcome! Please open an issue prior to submitting major changes.

---

## 📜 License
This project is licensed under the **MIT License** — feel free to use, modify, and distribute.

---

## 👨‍💻 Author
- **Om Jaikumar**  
- GitHub: [@omjaikumar23](https://github.com/omjaikumar23)  
- LinkedIn: [Om Jaikumar](https://www.linkedin.com/in/om-jaikumar-879410224/)

---

