# 🧠 TwinTalk: Real-Time ASL Translation with Digital Twin Integration

**TwinTalk** is a real-time American Sign Language (ASL) translation system that integrates AI-based gesture recognition with a digital twin for speech visualization. The goal of this project is to build an inclusive communication tool that can interpret sign language through webcam video and present the spoken equivalent using a realistic avatar.

## 🎯 Project Objective

We set out to develop a system capable of:
- Translating ASL gestures in real-time from webcam video
- Running efficiently on local hardware
- Outputting speech through a digital human avatar
- Training and evaluating deep learning models on a custom-built dataset

---

## ⚗️ Pre-Experiments and Decisions

We began with a **pre-trained I3D (Inflated 3D ConvNet)** model, inspired by the implementation used by the **WLASL (World’s Largest ASL) dataset**. However:

- ❌ The I3D model was **too large**, **slow to infer**, and **hard to implement** on local machines.
- ❌ Translation quality did **not meet our expectations**.
- ✅ We decided to **train lightweight models from scratch** for better speed, control, and accuracy.

---

## 🧪 Custom Dataset

We collected and labeled our own ASL dataset from scratch:

- **90 ASL classes**  
- **900 total videos** (10 videos per class)  
- ✅ No inner-class variation — for each word, we chose one consistent sign only  
  _(Most available datasets suffer from inner-class variation, where multiple signs represent the same word. This affects both training consistency and testing accuracy.)_
- 📦 [View the dataset on Kaggle](https://www.kaggle.com/datasets/jomanahalshangiti/twintalk-asl-data/data)

This consistency made it ideal for training controlled experiments with deep models.


---

## 🧠 Models Trained

We trained and evaluated two model architectures:

### 🧩 TwinTalk_with_CNN-LSTM
- A custom CNN + LSTM architecture
- Extracts spatial and temporal features from frame sequences
- Optimized for lower latency and simple training

### 🕸️ TwinTalk_with_ST-GCN
- Spatial-Temporal Graph Convolutional Network (ST-GCN)
- Works on **body keypoints** extracted from video
- Models temporal dynamics explicitly using pose graphs

---

## 🧍 Digital Twin Integration

We connected the recognized text output to a **digital twin avatar** using the [D-ID API](https://www.d-id.com/), enabling:

- Realistic speech animation
- Text-to-speech + lip sync
- Future integration with platforms like Zoom or MS Teams

---

## 📁 Repository Structure

```bash
TwinTalk/
├── TwinTalk_with_i3d/           # Initial I3D-based experiment (abandoned due to performance issues)
├── TwinTalk_with_ST-GCN/        # Graph-based gesture recognition using skeleton keypoints
├── TwinTalk_with_CNN-LSTM/      # CNN-LSTM model trained from scratch
└── README.md                    # You're here!
