# Pattern Recognition Project – Group 6

**Team:** Deepthika Sivaram & Derek Lu  
**Course:** Pattern Recognition — University at Buffalo  
**Live Demo:** <https://patternrec-project-group6.onrender.com/>

This project unifies two machine-learning pipelines—fruit image classification (Computer Vision) and tag-based recipe recommendation (Natural Language Processing)—into a single web application. The front-end runs real-time ONNX models in the browser, while a lightweight Flask API serves BERT-powered semantic search for recipes.

---

## Project Overview

| Phase | Goal | Key Models |
|-------|------|------------|
| **1. Data Collection** | 12 k smartphone images of four fruits (Apple, Grapes, Peach, Raspberry) in three photographic variants each. | – |
| **2. Computer Vision** | Transfer-learn **ResNet-50** (fruit type) and **EfficientNet-B0** (variant). | ResNet-50, EfficientNet-B0 |
| **3. NLP** | Semantic recipe retrieval from `RAW_recipes.csv` + `RAW_interactions.csv`. | Fine-tuned `all-MiniLM-L6-v2` https://huggingface.co/datasets/dsivaram/recipe-api/tree/main |
| **4. Deployment** | Single-page web app with ONNX.js (CV) + Flask (NLP) hosted on Render. | ONNX, Flask |

**Why it matters:** Grocers and meal-planning apps can instantly identify produce quality/variant **and** surface suitable recipes, creating a seamless “snap & cook” user journey.

---

## Quick Start

```bash
# 1. Clone repo
git clone https://github.com/deepthika-sivaram/PatternRec_Project_Group6.git
cd PatternRec_Project_Group6

# 2. Create & activate Python env (≥3.10 recommended)
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch Flask server (serves NLP + static front-end)
python app.py
