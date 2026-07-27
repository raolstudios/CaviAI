# 🦷 CaviAI — Deep Learning Dental Cavity Detector

**CaviAI** is an AI-powered proof-of-concept web application designed to assist in detecting dental caries (cavities) from cropped tooth radiograph patches. Powered by a fine-tuned **ResNet18** convolutional neural network, CaviAI analyzes localized X-ray regions and provides real-time diagnostic probabilities with enhanced sensitivity tuning to minimize false negatives.

---

## 📸 Key Features

* **Instant Inference:** Upload cropped dental radiograph patches (`PNG`, `JPG`, `JPEG`) for real-time analysis.
* **Sensitivity-Tuned Detection:** Utilizes a custom decision threshold ($35\%$) and class-weighted loss optimization during training to prioritize detecting subtle lesions and reduce false negatives.
* **Streamlit Web Interface:** A clean, intuitive dashboard designed for seamless interaction and visual confidence reporting.
* **Robust Preprocessing:** Handles image transformations and standardization dynamically using PyTorch's `torchvision` pipeline.

---

## 🏗️ Model Architecture & Methodology

* **Base Architecture:** ResNet18 (Pretrained on ImageNet, fine-tuned for binary classification).
* **Dataset:** Cropped tooth patches extracted from the **Zenodo Dental Cavity Dataset** via XML bounding box coordinates.
* **Data Augmentation:** Includes random rotations, affine transformations, horizontal flips, and brightness/contrast jittering (`ColorJitter`) to build resilience against varying X-ray exposure levels across different radiograph machines.
* **Loss Function Optimization:** Trained with weighted Cross-Entropy Loss ($3.0	imes$ penalty weight on cavity classes) to heavily penalize missing active caries lesions.
* **Validation Strategy:** Evaluated using an $80/20$ train/validation split on completely unseen radiograph patches to ensure genuine real-world performance.

---

## 🚀 Quickstart & Local Setup

### 1. Prerequisites
Ensure you have Python 3.9+ installed on your system.

### 2. Clone Repository & Navigate
```bash
git clone https://github.com/YOUR_USERNAME/CaviAI.git
cd CaviAI
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Model Weights Placement
Ensure your trained PyTorch weights file (`dental_cavity_model.pth`) is placed directly in the root directory alongside `app.py`.

### 5. Launch Application
```bash
streamlit run app.py
```

---

## 📂 Repository Structure

```text
CaviAI/
├── app.py                      # Main Streamlit application UI and inference script
├── dental_cavity_model.pth     # Fine-tuned PyTorch ResNet18 weight parameters
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation and setup guide
```

---

## ⚠️ Medical Disclaimer

> **Disclaimer:** CaviAI is developed strictly as a proof-of-concept software application for educational, research, and portfolio demonstration purposes. It is **not** a certified medical device and is **not** intended for clinical diagnosis, treatment planning, or real-world dental practice execution. All predictions must be independently evaluated by a licensed dental healthcare professional.

---

## 📜 Acknowledgments & Credits

* **Dataset:** [Zenodo Dental Cavity Dataset](https://zenodo.org/records/4907880)
* **Frameworks:** PyTorch, Torchvision, Streamlit, and Pillow.
