# Cotton Leaf Disease Classification (ConvNeXt-GCN-CLIP)

This project trains and deploys a ConvNeXt-GCN based CLIP model for cotton leaf disease classification, then serves predictions with a Streamlit app and RAG + Groq guidance.

## Project Structure

- app.py: Streamlit app
- scripts/download_dataset.py: Download dataset via gdown
- scripts/train.py: Train and export model weights
- models/: model weights
- knowledge_base/: management strategy docs (RAG)
- src/: model, data, RAG, and Groq helpers

## Setup

1. Create a virtual environment
2. Install requirements:

```bash
pip install -r requirements.txt
```

3. Create a `.env` file using `.env.example`:

```
GROQ_API_KEY=your_key
GROQ_MODEL=llama3-70b-8192
MODEL_WEIGHTS=f:/Thesis/models/convnext_gcn_clip_cotton_leaf.pth
DATASET_DIR=f:/Thesis/data/Cotton_Augmented_Dataset
KB_DIR=f:/Thesis/knowledge_base
```

## Download Dataset (gdown)

```bash
python scripts/download_dataset.py
```

## Train Model

```bash
python scripts/train.py --epochs 20 --batch-size 32
```

## Run Streamlit App

```bash
streamlit run app.py
```

## Notes

- If gdown fails, download the dataset zip manually and extract to `data/Cotton_Augmented_Dataset`.
- Update knowledge_base with your management strategy files (txt, md, pdf).
