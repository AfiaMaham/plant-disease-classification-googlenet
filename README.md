#  Bean Leaf Disease AI Classifier

A deep learning-based image classification system that detects diseases in bean leaves using a **GoogLeNet CNN model**, deployed with **Streamlit** for real-time predictions.

---

##  Features

-  Detects bean leaf diseases from images
-  Real-time prediction using Streamlit UI
-  Deep Learning model (GoogLeNet - Transfer Learning)
-  Softmax-based confidence scoring
-  CPU-compatible inference
-  Supports JPG, PNG, JPEG images

---

##  Model Details

- Architecture: **GoogLeNet (Inception v1)**
- Framework: **PyTorch**
- Transfer Learning: Pretrained on ImageNet
- Final Layer: Modified for custom classes

---

##  Dataset

Bean Leaf Lesions Classification Dataset (Kaggle)

### Classes:
- Healthy Leaf
- Angular Leaf Spot
- Bean Rust

---

##  Performance

- Training Accuracy: ~95%
- Test Accuracy: ~89%

---

##  Tech Stack

- Python 
- PyTorch 
- Torchvision
- Streamlit 
- NumPy
- PIL


