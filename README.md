# 🌌 Galaxy and Star Image Classifier using CNN

A deep learning project that classifies astronomical images into **stars** or **galaxies** using a **Convolutional Neural Network (CNN)** built with TensorFlow/Keras.

The model was trained on a dataset of over **3,986 grayscale images** (28x28 pixels) and achieved a validation accuracy of **~82.7%**, demonstrating strong performance for basic classification tasks in automated astronomy or educational use cases.

---

## 📁 Dataset

🔗 [Dummy Astronomy Data - Kaggle](https://www.kaggle.com/datasets/divyansh22/dummy-astronomy-data)

### Folder Structure:
```
archive(6)/Cutout Files/
│
├── star/
│   ├── star_image1.jpg
│   ├── star_image2.jpg
│   └── ...
│
└── galaxy/
    ├── galaxy_image1.jpg
    ├── galaxy_image2.jpg
    └── ...
```

This dataset was created as part of a project at the **Aryabhatta Research Institute of Observational Sciences (ARIES)**, Nainital, India. The images were captured using the institute’s in-house **1.3m telescope** located at the Devasthal observatory site.

Labeling was done using **image segmentation** and cross-referenced with the **Sloan Digital Sky Survey (SDSS) database**.

---

## 🧠 Model Architecture

We built a **Sequential CNN model** with the following layers:

- Input Layer: `(28, 28, 1)` for grayscale images  
- Conv2D Layers with MaxPooling and Dropout for feature extraction and regularization  
- Flatten Layer  
- Dense Hidden Layer with ReLU activation  
- Output Layer with Softmax activation for binary classification  

Total Trainable Parameters: **~392,098**

Optimizer: `Adam`  
Loss Function: `Sparse Categorical Crossentropy`  
Metrics: `Accuracy`

---

## 📊 Training Results

- **Training Accuracy**: ~86.8%
- **Validation Accuracy**: ~82.7%
- **Validation Loss**: ~0.3594

The model shows stable convergence with minimal overfitting due to dropout layers and proper data splitting.

---

## 🖼️ Sample Prediction

A sample image of a star was passed through the model, and it correctly predicted the class as `"Star"`.

---

## 🚀 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/galaxy-star-classifier.git
   ```

2. Install dependencies:
   ```bash
   pip install tensorflow numpy matplotlib opencv-python pillow
   ```

3. Download and extract the dataset from Kaggle into the `archive(6)/Cutout Files/` directory.

4. Run the Jupyter notebook or Python script:
   ```bash
   jupyter notebook galaxy_star_classifier.ipynb
   ```

---

## 📝 To Do / Future Improvements

- Add support for multi-class classification (e.g., quasars, nebulae)
- Increase image resolution and use transfer learning (e.g., MobileNet, ResNet)
- Deploy model as a web application using Flask or Streamlit
- Evaluate performance using precision, recall, and confusion matrix
---
🌟 Happy stargazing and happy coding!
