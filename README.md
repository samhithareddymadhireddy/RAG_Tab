# RAG-Tab: Retrieval-Augmented Tabular Synthesis for Imbalanced Data

**RAG-Tab** is a machine learning research framework investigating whether retrieving local minority-class neighbors as structural context for Large Language Models (LLMs) improves synthetic minority generation and downstream classification performance in imbalanced tabular data.

---

## 🔬 Core Research Question

> **Can local minority-class neighborhood retrieval provide useful structural and geometric context to an LLM, enabling it to synthesize high-fidelity minority instances that improve downstream classification while mitigating false-positive boundary degradation?**

---

## 📂 Project Structure

```text
RAG-Tab/
├── .env.example                     # API key configuration template
├── requirements.txt                 # Project dependencies
├── data/
│   ├── raw/
│   │   └── page-blocks0.dat         # KEEL dataset (5,472 samples, 10 numerical features, IR ~ 8.79)
│   ├── splits/                      # Leakage-free frozen train and test splits (80/20 stratified)
│   └── processed/                   # Generated synthetic tabular datasets
├── notebooks/
│   ├── 01_dataset_analysis.ipynb    # Phase A: EDA, missing value audit, stratified split, feature scaling
│   ├── 02_baselines.ipynb           # Phase B: 6 traditional baselines (ROS, SMOTE unscaled/scaled, Borderline-SMOTE, ADASYN, SMOTE-Tomek)
│   ├── 03_retrieval.ipynb           # Phase D: Local minority neighborhood retrieval (KNN & Natural Neighbors)
│   ├── 04_llm_generation.ipynb      # Phase E: Unguided LLM-only generation baseline (Gemini Flash)
│   ├── 05_rag_tab.ipynb             # Phase F & G: RAG-Tab end-to-end pipeline with multi-stage quality control
│   └── 06_evaluation.ipynb          # Phase H & K: Cross-classifier evaluation, statistical fidelity, PCA & t-SNE diagnostics
├── src/
│   ├── baselines/                   # Standardized resampling implementations with scaling & provenance tracking
│   ├── data/                        # KEEL loader and frozen train/test split manager
│   ├── evaluation/                  # Precision, Recall, F1, Specificity, G-Mean, Wasserstein distance, KS-tests
│   ├── generation/                  # Prompt builder, Gemini Flash client, and JSON schema validator
│   ├── retrieval/                   # Distance-based KNN & Natural Neighbor retriever
│   └── validation/                  # Multi-stage synthetic quality controller
└── results/
    ├── plots/                       # High-resolution benchmark charts and manifold projections
    └── tables/                      # Experimental CSV tables and leaderboards
```

---

## 📊 Summary of Experimental Results (`page-blocks0`)

Tested on held-out test split ($N=1,095$, Minority $N=112$, Majority $N=983$) using `RandomForestClassifier(n_estimators=300, random_state=42)`:

| Method | Train Rows | Added Rows | Precision | Recall | F1-Score | Specificity | G-Mean | False Positives |
|---|---|---|---|---|---|---|---|---|
| **Original (No Resampling)** | 4,377 | 0 | **0.9099** | 0.9018 | 0.9058 | **0.9898** | 0.9448 | **10** |
| **Random Oversampling (ROS)**| 7,860 | 3,483 (dup) | 0.8750 | 0.9375 | 0.9052 | 0.9847 | 0.9608 | 15 |
| **SMOTE (Unscaled)** | 7,860 | 3,483 (synth) | 0.8182 | 0.9643 | 0.8852 | 0.9756 | 0.9699 | 24 |
| **SMOTE (Standardized)** | 7,860 | 3,483 (synth) | 0.8438 | 0.9643 | 0.9000 | 0.9797 | 0.9719 | 20 |
| **Borderline-SMOTE** | 7,860 | 3,483 (synth) | 0.8321 | **0.9732** | 0.8971 | 0.9776 | **0.9754** | 22 |
| **ADASYN** | 7,915 | 3,538 (synth) | 0.7899 | **0.9732** | 0.8720 | 0.9705 | 0.9719 | 29 |
| **SMOTE-Tomek** | 7,820 | 3,443 (clean) | 0.8182 | 0.9643 | 0.8852 | 0.9756 | 0.9699 | 24 |
| **LLM-Only (Unguided)** | 4,427 | 50 (synth) | 0.9043 | 0.9286 | **0.9163** | 0.9888 | 0.9582 | 11 |
| **RAG-Tab (Proposed)** | 4,427 | 50 (synth) | 0.9035 | 0.9196 | **0.9115** | 0.9888 | 0.9536 | 11 |

---

## 🚀 Setup & Execution

### 1. Install Dependencies
```bash
pip install -r requirements.txt
pip install google-genai python-dotenv
```

### 2. Configure Gemini API Key (Optional for Live API)
Copy `.env.example` to `.env` and set your Google AI Studio API key:
```bash
cp .env.example .env
```
*(If no API key is set, the pipeline automatically operates in validation mock mode to allow offline testing.)*

### 3. Run the Research Notebooks in Order:
1. `notebooks/01_dataset_analysis.ipynb` — Data loading and frozen split generation.
2. `notebooks/02_baselines.ipynb` — Traditional oversampling benchmarks.
3. `notebooks/03_retrieval.ipynb` — Local neighborhood retrieval diagnostics.
4. `notebooks/04_llm_generation.ipynb` — Unguided LLM baseline.
5. `notebooks/05_rag_tab.ipynb` — End-to-end RAG-Tab with quality control.
6. `notebooks/06_evaluation.ipynb` — Cross-classifier benchmarks & statistical diagnostics.
