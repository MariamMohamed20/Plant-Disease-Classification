# Plant-Disease-Classification
# 🌿 Plant Disease Classification using Deep Learning

A deep learning project that automatically detects and classifies plant diseases from leaf images using three pretrained CNN architectures, with explainability via Grad-CAM and a Streamlit GUI for real-time predictions.

---


## 📌 Problem Statement
Plant diseases cause significant agricultural losses worldwide. Early and accurate detection is critical for farmers and food security. This project leverages transfer learning to classify 38 plant disease categories from leaf images — making expert-level diagnosis fast and accessible.

---

## 📂 Dataset
- **Source:** [PlantVillage Dataset (Kaggle)](https://www.kaggle.com/datasets/emmarex/plantdisease)
- **Images:** 54,000+ labeled leaf images
- **Classes:** 38 disease categories across multiple plant species
- **Split:** 70–80% training / 10–15% validation / 10–15% test

---

## 🧠 Models Used

| Model | Architecture | Pretrained On | Key Strength |
|---|---|---|---|
| ResNet50 | 50-layer residual network | ImageNet | Strong baseline, well-tested |
| EfficientNetB3 | Compound-scaled CNN | ImageNet | High accuracy, efficient |
| MobileNetV2 | Lightweight depthwise CNN | ImageNet | Fast inference, mobile-friendly |

All models use **transfer learning** — pretrained ImageNet weights with a modified final layer for 38 disease classes.

---

## ⚙️ Methodology

### Data Preprocessing
- Resized all images to 224×224
- Normalized pixel values (ImageNet normalization)
- Removed corrupted images and verified labels
- Handled class imbalance

### Data Augmentation
- Random rotation (±15–30°)
- Horizontal/vertical flip
- Brightness and contrast adjustment
- Zoom and shift
- Color jittering
- **Ablation study:** trained with and without augmentation to measure impact

### Training Strategy
- Loaded pretrained weights (ImageNet)
- Experimented with learning rates: 1e-3, 1e-4, 1e-5
- Tested batch sizes: 16, 32, 64
- Compared optimizers: Adam vs SGD
- Applied early stopping and saved best model weights
- Experimented with different layer freezing strategies

---

## 📊 Results

| Metric | ResNet50 | EfficientNetB3 | MobileNetV2 |
|---|---|---|---|
| Accuracy | — | — | — |
| Precision | — | — | — |
| Recall | — | — | — |
| F1-Score | — | — | — |
| Training Time | — | — | — |
| Model Size | — | — | — |
| Inference Speed | — | — | — |

> 📌 Results will be updated upon completion of all model training.

---

## 🔍 Explainability — Grad-CAM
We use **Gradient-weighted Class Activation Mapping (Grad-CAM)** to visualize what each model focuses on when making predictions.

- 5 correctly classified images per model
- 5 incorrectly classified images per model
- Total: 30 annotated heatmap visualizations
- Reveals differences in how ResNet50, EfficientNetB3, and MobileNetV2 interpret leaf features

---

## 🖥️ GUI Application (Streamlit)

A three-page interactive web application:

**Page 1 — Prediction Interface**
- Upload a leaf image (JPG/PNG) or use webcam
- Select model (ResNet50 / EfficientNetB3 / MobileNetV2)
- View top-3 predicted diseases with confidence scores
- See Grad-CAM heatmap overlay

**Page 2 — Performance Dashboard**
- Accuracy comparison charts
- Confusion matrices for all three models
- Per-disease performance breakdown
- Training curves

**Page 3 — About**
- Project description, dataset info, team roles, usage guide

---

## 📱 Bonus: TensorFlow Lite Deployment (+5 pts)
- Converted MobileNetV2 to TFLite format
- Applied quantization for mobile optimization
- Benchmarked: original vs TFLite size, speed, and accuracy

---

## 🚀 How to Run

### Prerequisites
```bash
pip install -r requirements.txt
```

### Run the Streamlit App
```bash
streamlit run gui/app.py
```


## 🛠️ Technologies
Python · TensorFlow · Keras · Streamlit · OpenCV · scikit-learn · NumPy · Matplotlib · Google Colab · Kaggle

---

## 📚 References
- [PlantVillage Dataset](https://www.kaggle.com/datasets/emmarex/plantdisease)
- He et al. (2016) — Deep Residual Learning for Image Recognition
- Tan & Le (2019) — EfficientNet: Rethinking Model Scaling for CNNs
- Howard et al. (2018) — MobileNetV2
