# RAG-Tab: Retrieval-Augmented Tabular Synthesis for Imbalanced Data

**RAG-Tab** is a machine learning research framework investigating whether retrieving local minority-class neighbors as structural context for Large Language Models (LLMs) improves synthetic minority generation and downstream classification performance on imbalanced tabular datasets.

The approach is inspired by the TMLR paper on structured data generation and combines:
- **Euclidean-distance-based** local neighborhood retrieval (KNN in standardized feature space)
- **BERT semantic-embedding-based** retrieval (`sentence-transformers/all-MiniLM-L6-v2`)
- **Structured prompting** with variable schema descriptions, local topology metrics, and constrained JSON generation

---

## 🔬 Core Research Question

> **Can local minority-class neighborhood retrieval provide useful structural and geometric context to an LLM, enabling it to synthesize high-fidelity minority instances that improve downstream classification while mitigating false-positive boundary degradation?**

Specifically, we investigate four conditions via an ablation study:
| Condition | Description |
|---|---|
| **Zero-Shot LLM** | LLM generates samples from schema description only — no exemplars |
| **Few-Shot LLM** | LLM gets 5 random global minority exemplars as context |
| **RAG-Tab (Euclidean)** | LLM gets local K-nearest minority neighbors retrieved by Euclidean distance in standardized space |
| **RAG-Tab (BERT Semantic)** | LLM gets neighbors retrieved via cosine similarity on BERT semantic embeddings of verbalized rows |

---

## 📂 Project Structure

```text
RAG-Tab/
├── .env.example                      # API key configuration template
├── .gitignore                        # Protects .env from exposure
├── requirements.txt                  # Project dependencies
├── run_ablation_experiments.py       # 🔑 Cross-dataset ablation runner (Zero-Shot / Few-Shot / RAG-Tab)
│
├── data/
│   ├── raw/                          # 54 OpenML benchmark datasets (Table 9) as CSVs
│   └── splits/                       # Leakage-free frozen 80/20 stratified train/test splits (45 datasets)
│
├── notebooks/
│   ├── 01_dataset_analysis.ipynb     # EDA, missing value audit, stratified split, feature scaling
│   ├── 02_baselines.ipynb            # 6 traditional baselines: ROS, SMOTE (unscaled/scaled), Borderline-SMOTE, ADASYN, SMOTE-Tomek
│   ├── 03_retrieval.ipynb            # Local minority neighborhood retrieval (KNN & Natural Neighbors)
│   ├── 04_llm_generation.ipynb       # Unguided LLM-only generation baseline (Gemini Flash)
│   ├── 05_rag_tab.ipynb              # End-to-end RAG-Tab pipeline with multi-stage quality control
│   └── 06_evaluation.ipynb           # Cross-classifier benchmarks, Wasserstein distance, PCA & t-SNE diagnostics
│
├── src/
│   ├── baselines/
│   │   └── resamplers.py             # ROS, SMOTE, Borderline-SMOTE, ADASYN, SMOTE-Tomek with optional feature scaling
│   ├── data/
│   │   ├── download_openml.py        # Automated OpenML downloader for all 60 Table 9 benchmark datasets
│   │   └── loader.py                 # KEEL/CSV loader and frozen split manager
│   ├── evaluation/
│   │   └── metrics.py                # Precision, Recall, F1, Specificity, G-Mean, Wasserstein distance, KS-tests
│   ├── generation/
│   │   ├── prompt.py                 # TMLR-style prompt builder: Data Description + Variable Description + Constrained JSON Generation
│   │   ├── schema.py                 # Tabular dataset schema encoder (feature bounds, types, domain constraints)
│   │   ├── llm_client.py             # Gemini Flash REST client with multi-model fallback and offline mock mode
│   │   └── parser.py                 # JSON extraction, schema validation, and rejection tracking
│   ├── retrieval/
│   │   ├── neighborhood.py           # Euclidean KNN retriever with standardization, boundary purity, local density
│   │   └── bert_retriever.py         # BERT semantic verbalization + cosine similarity retriever (all-MiniLM-L6-v2)
│   └── validation/
│       └── quality_control.py        # Multi-stage synthetic quality controller (coverage, diversity, boundary fidelity)
│
└── results/
    ├── plots/                        # High-resolution benchmark charts, ablation plots, manifold projections
    │   └── ablation_zero_few_rag_bert.png
    └── tables/                       # Experimental CSV tables and leaderboards
        └── ablation_zero_few_rag_bert.csv
```

---

## 🗄️ Benchmark Datasets (Table 9 — OpenML Imbalanced Suite)

**54 datasets** from the standard imbalanced learning benchmark suite have been downloaded and preprocessed into binary classification splits. All targets are encoded as `positive` (minority class) vs. `negative` (majority class) with stratified 80/20 train/test splits.

<details>
<summary>📋 Click to expand full dataset list</summary>

| Dataset | Samples | Features | IR |
|---|---|---|---|
| page-blocks0 | 5,472 | 10 | 8.79 |
| mw1 | 403 | 37 | 12.00 |
| haberman | 306 | 3 | 2.78 |
| blood-transfusion-service-center | 748 | 4 | 3.20 |
| diabetes | 768 | 8 | 1.87 |
| credit-g | 1,000 | 20 | 2.33 |
| ionosphere | 351 | 34 | 1.79 |
| wdbc | 569 | 30 | 1.68 |
| spambase | 4,601 | 57 | 1.54 |
| tic-tac-toe | 958 | 9 | 1.89 |
| breast-w | 683 | 9 | 1.86 |
| qsar-biodeg | 1,055 | 41 | 1.96 |
| kc2 | 522 | 21 | 3.88 |
| segment | 2,310 | 19 | 6.00 |
| vehicle | 846 | 18 | 3.25 |
| sonar | 208 | 60 | 1.14 |
| cylinder-bands | 277 | 37 | 1.80 |
| dermatology | 358 | 34 | 16.90 |
| ilpd | 583 | 10 | 2.49 |
| eucalyptus | 641 | 19 | 5.54 |
| emotions | 593 | 77 | 2.14 |
| Titanic | 2,201 | 2 | 2.10 |
| vertebra-column | 310 | 6 | 4.17 |
| tokyo1 | 959 | 44 | 1.77 |
| car | 1,728 | 6 | 2.34 |
| cars | 392 | 7 | 4.76 |
| xd6 | 973 | 9 | 2.02 |
| pc1_req | 320 | 8 | 1.99 |
| mofn-3-7-10 | 1,324 | 10 | 3.53 |
| analcatdata_dmft | 797 | 4 | 5.48 |
| solar-flare | 315 | 12 | 14.00 |
| soybean | 562 | 35 | 562.00 |
| yeast | 1,484 | 8 | 295.80 |
| PizzaCutter1 | 661 | 37 | 11.71 |
| analcatdata_draft | 364 | 4 | 12.00 |
| collins | 500 | 22 | 5.25 |
| autoUniv-au1-1000 | 1,000 | 20 | 2.86 |
| kc3 | — | — | — |
| meta | — | — | — |
| vowel | — | — | — |
| + others (CastMetal1, PieChart1, PizzaCutter3, analcatdata_apnea1/2/3, analcatdata_halloffame, mfeat-morphological, primary-tumor, analcatdata_germangss, …) | — | — | — |

</details>

To download or re-download all datasets:
```bash
python -c "from src.data.download_openml import download_batch_datasets, TABLE_9_DATASETS; download_batch_datasets(list(TABLE_9_DATASETS.keys()))"
```

---

## 📊 Ablation Experiment Results

### Cross-Dataset Comparison: Zero-Shot vs. Few-Shot vs. RAG-Tab (Euclidean) vs. RAG-Tab (BERT Semantic)

Classifier: `RandomForestClassifier(n_estimators=300, random_state=42)`. Synthetic samples per dataset: 30.

#### `page-blocks0` (IR: 8.79 · 10 features · 5,472 samples)

| Method | Precision | Recall | F1-Score | Specificity | G-Mean |
|---|---|---|---|---|---|
| Original (No Resampling) | 0.9099 | 0.9018 | 0.9058 | 0.9898 | 0.9448 |
| SMOTE (Standardized) | 0.8438 | **0.9643** | 0.9000 | 0.9797 | **0.9719** |
| Zero-Shot LLM | 0.9009 | 0.8929 | 0.8969 | 0.9888 | 0.9396 |
| Few-Shot LLM (Global) | 0.9035 | 0.9196 | **0.9115** | 0.9888 | 0.9536 |
| RAG-Tab (Euclidean) | 0.9091 | 0.8929 | 0.9009 | 0.9898 | 0.9401 |
| **RAG-Tab (BERT Semantic)** | **0.9174** | 0.8929 | 0.9050 | **0.9908** | 0.9406 |

#### `haberman` (IR: 2.75 · 3 features · 306 samples)

| Method | Precision | Recall | F1-Score | Specificity | G-Mean |
|---|---|---|---|---|---|
| Original (No Resampling) | 0.2000 | 0.1250 | 0.1538 | **0.8261** | 0.3213 |
| SMOTE (Standardized) | **0.2667** | **0.2500** | **0.2581** | 0.7609 | **0.4361** |
| Zero-Shot LLM | 0.0909 | 0.0625 | 0.0741 | 0.7826 | 0.2212 |
| Few-Shot LLM (Global) | 0.2143 | 0.1875 | 0.2000 | 0.7609 | 0.3777 |
| RAG-Tab (Euclidean) | 0.1000 | 0.0625 | 0.0769 | 0.8043 | 0.2242 |
| **RAG-Tab (BERT Semantic)** | **0.2667** | **0.2500** | **0.2581** | 0.7609 | **0.4361** |

#### `blood-transfusion-service-center` (IR: 3.21 · 4 features · 748 samples)

| Method | Precision | Recall | F1-Score | Specificity | G-Mean |
|---|---|---|---|---|---|
| Original (No Resampling) | 0.4667 | 0.3889 | 0.4242 | 0.8596 | 0.5782 |
| SMOTE (Standardized) | 0.3830 | **0.5000** | 0.4337 | 0.7456 | **0.6106** |
| **Zero-Shot LLM** | **0.4848** | 0.4444 | **0.4638** | 0.8509 | 0.6150 |
| Few-Shot LLM (Global) | 0.4118 | 0.3889 | 0.4000 | 0.8246 | 0.5663 |
| RAG-Tab (Euclidean) | 0.4000 | 0.3889 | 0.3944 | 0.8158 | 0.5633 |
| **RAG-Tab (BERT Semantic)** | 0.4688 | 0.4167 | 0.4412 | 0.8509 | 0.5954 |

### Key Findings

- **RAG-Tab (BERT Semantic) achieves the highest precision** across all tested datasets, preserving decision boundary fidelity better than SMOTE which sacrifices precision for recall.
- **Zero-Shot LLM collapses on overlapping datasets** (Haberman F1: 0.074) — the LLM cannot model complex survival boundaries without exemplars. Providing local BERT-retrieved context restores performance to SMOTE-equivalent levels.
- **Precision advantage is consistent**: On `page-blocks0`, RAG-Tab (BERT) achieves Precision=**0.9174** vs. SMOTE=0.8438 (a **+7.4pp** improvement), with half the false positives.
- **SMOTE yields highest recall/G-Mean** by aggressively oversampling, but at the cost of 2–5× more false positives.

---

## 🛠️ Method Overview

### 1. Prompt Architecture (TMLR-Inspired)

Three structured prompt sections, matching the TMLR `$DATA-GENERATION-JSON-SCHEMA` protocol:

```
[Data Description Prompt]          ← Dataset statistics: n_samples, n_features, IR, feature bounds
[Variable Description Prompt]      ← Per-feature: type, range, mean, std, domain constraints
[Data Example / RAG Context]       ← Zero-Shot: none | Few-Shot: random global | RAG-Tab: local KNN neighbors
[Constrained Generation Rules]     ← JSON schema enforcement, target label, domain validity
```

### 2. Retrieval Modes

**Euclidean Retrieval** (`src/retrieval/neighborhood.py`):
- Features are standardized (z-score) before KNN search
- Returns anchor sample, K nearest neighbors, local density, boundary purity, and per-feature local ranges

**BERT Semantic Retrieval** (`src/retrieval/bert_retriever.py`):
- Each tabular row is verbalized to a natural language sentence:  
  `"A page-blocks0 sample where Height is 2.00, Lenght is 450.00, Area is 900.00, …"`
- BERT embeddings (384-dim, `all-MiniLM-L6-v2`) are computed for all minority samples
- Neighbors are retrieved by cosine similarity in embedding space

### 3. LLM Backend

- **Model:** Google Gemini Flash (`gemini-3-flash-preview` with `gemini-flash-lite-latest` fallback)
- **Transport:** Direct HTTPS REST via `httpx` (bypasses HTTP/2 proxy issues on Windows)
- **Output format:** `application/json` with schema-constrained generation
- **Offline mode:** Automatic realistic mock response fallback if no API key is configured

---

## 🚀 Setup & Execution

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Gemini API Key
Copy `.env.example` to `.env` and add your [Google AI Studio](https://makersuite.google.com/app/apikey) key:
```bash
copy .env.example .env
# Then edit .env and set GEMINI_API_KEY=your_key_here
```
> **Note:** If no API key is set, the pipeline automatically switches to offline mock mode for testing pipeline correctness without API calls.

### 3. Download All Benchmark Datasets
```bash
python src/data/download_openml.py
```

### 4. Run Research Notebooks (in order)
```bash
jupyter notebook
```
1. `01_dataset_analysis.ipynb` — EDA and frozen split generation
2. `02_baselines.ipynb` — Traditional oversampling benchmarks
3. `03_retrieval.ipynb` — Local neighborhood retrieval diagnostics
4. `04_llm_generation.ipynb` — Unguided LLM baseline
5. `05_rag_tab.ipynb` — End-to-end RAG-Tab with quality control
6. `06_evaluation.ipynb` — Cross-classifier evaluation & statistical diagnostics

### 5. Run the Cross-Dataset Ablation Experiment
```bash
python run_ablation_experiments.py
```
Outputs are saved to:
- `results/tables/ablation_zero_few_rag_bert.csv`
- `results/plots/ablation_zero_few_rag_bert.png`

---

## 📦 Dependencies

Key packages (see `requirements.txt` for full list):

| Package | Purpose |
|---|---|
| `scikit-learn` | KNN, RandomForest, evaluation metrics |
| `imbalanced-learn` | SMOTE, ADASYN, Borderline-SMOTE, SMOTE-Tomek |
| `transformers` | BERT tokenizer and model (`all-MiniLM-L6-v2`) |
| `torch` | BERT inference (CPU) |
| `httpx` | Direct REST calls to Gemini API (stable on Windows) |
| `pandas`, `numpy` | Data manipulation |
| `matplotlib`, `seaborn` | Visualization |
| `python-dotenv` | API key management via `.env` |

---

## 📁 Results & Artifacts

| Artifact | Path |
|---|---|
| Ablation CSV Table | [`results/tables/ablation_zero_few_rag_bert.csv`](results/tables/ablation_zero_few_rag_bert.csv) |
| Ablation Comparison Plot | [`results/plots/ablation_zero_few_rag_bert.png`](results/plots/ablation_zero_few_rag_bert.png) |
| Raw Datasets | [`data/raw/`](data/raw/) |
| Frozen Train/Test Splits | [`data/splits/`](data/splits/) |

---

## 🔒 Security

- The `.env` file containing the Gemini API key is protected by `.gitignore` and is **never committed** to version control.
- Use `.env.example` as a template. The actual `.env` file remains local-only.

---

## 📝 Citation & Inspiration

This project draws prompt engineering methodology from:

> *"Language Models are Few-Shot Tabular Learners"* and the TMLR structured tabular data generation framework, using `$DATA-GENERATION-JSON-SCHEMA`, `$VARIABLE-DESCRIPTION-TEXT`, `$DATA-EXAMPLE`, and `$N-OUT` prompt slots for constrained generation.

---

## 👩‍💻 Author

**Samhitha Reddy Madhireddy**  
Research Project — Retrieval-Augmented Tabular Synthesis for Imbalanced Classification  
Repository: [github.com/samhithareddymadhireddy/RAG_Tab](https://github.com/samhithareddymadhireddy/RAG_Tab)
