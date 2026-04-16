# 🌿 Cotton Leaf Disease Classification (Streamlit App)

This project is a deep learning–based web application for **cotton leaf disease classification** using a trained **ConvNeXt-GCN + CLIP model**.  
Users can upload a cotton leaf image and instantly get the predicted disease along with confidence scores and attention visualization.

---

## 🚀 Features

- 🌿 Image-based cotton leaf disease classification  
- 🤖 Pre-trained ConvNeXt-GCN + CLIP model  
- 📊 Class probability scores  
- 🔥 Attention map visualization  
- ⚡ Fast inference (no training required)  
- ☁️ Deployable on Streamlit Cloud  

---

## 🧠 Model

- Architecture: **ConvNeXt + GCN + CLIP**
- Input size: `224 × 224`
- Output classes:
  - Alternaria Leaf Spot  
  - Bacterial Blight  
  - Fusarium Wilt  
  - Healthy Leaf  
  - Verticillium Wilt  

---

## 📂 Project Structure

```
project/
│
├── app.py              # Streamlit UI
├── config.py           # Configurations (paths, model)
├── predict.py          # Model loading & inference
├── prompts.py          # Class names & labels
├── requirements.txt    # Dependencies
│
└── src/
    ├── modeling.py     # Model architecture
    └── data.py         # Image preprocessing
```

## ☁️ Deployment (Streamlit Cloud)
```
Upload this project to GitHub
The model is automatically downloaded from Google Drive
Make sure:
Model link is public (Anyone with the link)
requirements.txt is correct
```
## 📤 Usage
```
Upload a cotton leaf image
The model predicts the disease
View:
✅ Predicted disease name
📊 Confidence scores
🔥 Attention heatmap
```
