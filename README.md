# 🌾 CropLogic AI

**An Integrated Agricultural Decision-Support System for Maize Yield Prediction, Pest Detection, Soil Assessment, and RAG-Based Advisory**

> Final Year BSc Project — The Knowledge Hub Universities (Coventry University)  
> Author: Youssef Mohamed Hanafy | Student ID: YH202000009  
> Supervisor: Dr. Nada Gaballah

---

## 📌 Overview

CropLogic AI is a fully implemented, modular agricultural intelligence platform that combines four machine-learning components into a single unified framework. Using US county-level maize data as a case study, the system spans yield prediction, soil health assessment, pest and disease detection, and a retrieval-augmented generation (RAG) advisory chatbot — all accessible through an interactive Streamlit web application.

The system was designed to address a key gap in agricultural AI: most existing tools operate in isolation. CropLogic AI integrates outputs across all modules so that soil findings inform yield predictions, pest results feed the knowledge base, and the chatbot grounds its answers in the system's own data.

---

## 🗂️ Repository Structure

```
CropLogic-AI/
│
├── CropAI-01-Preprocessing.ipynb     # Data preprocessing & feature engineering
├── CropAI-02-Models.ipynb            # Yield prediction ensemble
├── CropAI-03-Soil.ipynb              # Soil health assessment
├── CropAI-04-PestDetection.ipynb     # CNN pest/disease classification
├── CropAI-05 edited.ipynb            # RAG advisory chatbot
│
├── app.py                            # Streamlit web application (5 pages)
│
├── usa_maize_county_level_1961_2024.csv   # Raw dataset (62,400 rows)
├── usa_maize_preprocessed.csv             # After preprocessing (19 columns)
├── usa_maize_features.csv                 # After feature engineering (69 columns)
├── soil_assessment.csv                    # Soil module output (975 counties)
│
└── README.md
```

---

## 🧩 Modules

### CropAI-01 — Data Preprocessing & Feature Engineering
Processes the raw county-level maize dataset into a model-ready feature matrix.

- **Input:** `usa_maize_county_level_1961_2024.csv` — 62,400 observations, 975 counties, 15 US states, 1961–2024
- **Steps:** Value clipping, silt derivation, USDA texture classification, ordinal state encoding, time-based features
- **Feature groups engineered:** Climate transforms, county anomalies, soil composites, spatial features, time trends, interaction terms, lag & rolling features, stress indices
- **Output:** `usa_maize_features.csv` — 62,400 rows × 69 columns

---

### CropAI-02 — Yield Prediction Ensemble
Trains and evaluates a weighted ensemble of four regression models.

| Model | Temporal R² | Random R² |
|---|---|---|
| Ridge Regression | 0.0150 | 0.7520 |
| Random Forest | 0.4959 | 0.9494 |
| Extra Trees | 0.4997 | 0.9534 |
| Gradient Boosting | 0.5177 | 0.9312 |
| **Ensemble** | **0.5239** | **0.9497** |

- **Temporal split:** Train 1961–2009, test 2010–2024 (primary evaluation)
- **Top features:** `yield_lag1` (0.300), `trend_sq` (0.207), `trend_linear` (0.201)
- **Output:** Saved model bundle (`agriAI_models.pkl`)

---

### CropAI-03 — Soil Health Assessment
Scores and ranks all 975 counties on a 0–100 soil health scale.

- **Method:** 64-year average per county → constraint flagging → composite score (pH 33% + SOC 33% + texture 34%)
- **Tiers:** Good (≥70) · Moderate (50–69) · Poor (<50)
- **Top state:** Iowa (89.1/100) — all 94 counties rated Good
- **Most common constraint:** Low SOC (<1.5%) — 90 counties (9.2%)
- **Output:** `soil_assessment.csv` — 975 rows × 16 columns

---

### CropAI-04 — Pest & Disease Detection
Classifies maize leaf images into four disease categories using transfer learning.

| Class | Validation Accuracy | Images |
|---|---|---|
| Blight | 85% | 229 |
| Common Rust | 100% | 261 |
| Gray Leaf Spot | 91% | 114 |
| Healthy | 100% | 232 |
| **Overall** | **94.50%** | **836** |

- **Architecture:** EfficientNetB0 (ImageNet pre-trained) + Global Average Pooling + Dropout(30%) + Softmax(4)
- **Training:** 2-phase (frozen base → full fine-tune), EarlyStopping, ReduceLROnPlateau
- **Dataset:** PlantVillage maize leaf images (4,188 total)

---

### CropAI-05 — RAG Advisory Chatbot
A retrieval-augmented generation chatbot grounded in the system's own outputs.

- **Knowledge base:** 38 documents — 7 yield + 11 soil + 8 pest + 12 agronomic facts
- **Retrieval:** TF-IDF vectorisation (unigrams + bigrams, 938 terms) + cosine similarity
- **Top-5 documents** retrieved per query with relevance threshold 0.05
- **Optional:** OpenAI GPT integration for fluent natural language responses
- **Demo queries:** Yield model accuracy · State soil health · Pest identification · Disease management · Climate effects

---

## 🖥️ Streamlit Web Application

The `app.py` file runs a 5-page interactive dashboard:

| Page | Description |
|---|---|
| 🏠 Home | Project overview and module navigation |
| 📈 Yield Prediction | Input climate/soil features and get a yield forecast |
| 🌱 Soil Assessment | Explore county and state soil health scores |
| 🐛 Pest Detection | Upload a maize leaf image for disease classification |
| 💬 RAG Chatbot | Ask agronomic questions — buttons trigger live answers |

### Run locally

```bash
# 1. Clone the repository
git clone https://github.com/youssefmohanafy/CropLogic-AI.git
cd CropLogic-AI

# 2. Install dependencies
pip install streamlit scikit-learn pandas numpy matplotlib tensorflow pillow openai

# 3. Run the app
streamlit run app.py
```

> **Note:** The pest detection page requires the saved model file (`maize_disease_model.h5`).  
> The RAG chatbot requires `rag_knowledge_base.json` and `rag_vectorizer.pkl` generated by CropAI-05.

### 📓 Run on OneDrive (pre-configured environment)

All five notebooks are available in a pre-configured OneDrive environment with datasets and dependencies already set up:

**📂 [CL Notebook — OneDrive](https://elsewedyedu1-my.sharepoint.com/:f:/g/personal/yh2000009_tkh_edu_eg/IgCDGj8ffE14QLr_KyloHaCPAZ6A95dHkbnPD0PwDQU4qEQ?e=ftU0ma)**

> Open the notebooks directly from OneDrive without any local setup required.

---

## 📦 Datasets

| Dataset | Source | Size | Used In |
|---|---|---|---|
| US County-Level Maize (1961–2024) | Included in repo | 62,400 rows × 13 cols | CropAI-01, 02, 03 |
| Corn / Maize Leaf Disease Images | [Kaggle ↗](https://www.kaggle.com/datasets/smaranjitghose/corn-or-maize-leaf-disease-dataset) | 4,188 images | CropAI-04 |
| RAG Knowledge Base | Generated from module outputs | 38 documents | CropAI-05 |

### 🌿 Setting up the Pest Detection Dataset (CropAI-04)

The maize leaf images are **not included in this repo** due to size. Download them from Kaggle:

**📥 https://www.kaggle.com/datasets/smaranjitghose/corn-or-maize-leaf-disease-dataset**

After downloading, organise the images into the following folder structure in the root of the project:

```
PestDetection/
├── Blight/           # 1,146 images — Northern Corn Leaf Blight
├── Common_Rust/      # 1,306 images — Common Rust (Puccinia sorghi)
├── Gray_Leaf_Spot/   # 574  images — Gray Leaf Spot (Cercospora zeae-maydis)
└── Healthy/          # 1,162 images — Healthy maize leaves
```

> **Note:** The Kaggle dataset folder names may differ slightly. Rename them to match the structure above exactly, as `CropAI-04-PestDetection.ipynb` expects the `PestDetection/` directory with these four subfolder names.

Original academic source: Hughes, D. & Salathé, M. (2015). An open access repository of images on plant health to enable the development of mobile disease diagnostics. *arXiv:1511.08060*.

---

## 📊 Key Results Summary

| Module | Metric | Value |
|---|---|---|
| Yield Prediction | Temporal R² (2010–2024) | 0.52 |
| Yield Prediction | Random split R² | 0.95 |
| Yield Prediction | Temporal RMSE | 1,096 kg/ha |
| Pest Detection | Validation accuracy | 94.50% |
| Soil Assessment | Top state (Iowa) score | 89.1 / 100 |
| Soil Assessment | Counties with no constraints | 837 / 975 (85.8%) |
| RAG Chatbot | Query types answered correctly | 6 / 6 |

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.11 |
| ML / Data | scikit-learn, pandas, numpy |
| Deep Learning | TensorFlow, Keras (EfficientNetB0) |
| NLP / RAG | TF-IDF (scikit-learn), OpenAI API |
| Visualisation | Matplotlib |
| Web App | Streamlit |
| Environment | Jupyter Notebooks |

---

## 📁 Generated Files (not in repo — run notebooks to generate)

| File | Generated by | Description |
|---|---|---|
| `agriAI_models.pkl` | CropAI-02 | Saved ensemble model bundle |
| `maize_disease_model.h5` | CropAI-04 | Trained EfficientNetB0 weights |
| `rag_knowledge_base.json` | CropAI-05 | 38-document knowledge base |
| `rag_vectorizer.pkl` | CropAI-05 | Fitted TF-IDF vectorizer |

---

## 📄 License

This project was developed as an academic capstone submission. All rights belong to the author and The Knowledge Hub Universities (Coventry University).
