# Publication Readiness Audit & Paper Wrap-Up Report

**Repository:** `C:\Projects\ML\RAG_Tab`  
**Git Branch:** `audit/rag-tab-publication-readiness`  
**Paper Version:** `main_v2.tex` (Single-Column Springer Nature Format, `sn-jnl.cls`)  
**Backup:** `main_v1_backup.tex`

---

## 1. Number Traceability Matrix (`main_v2.tex`)

Every quantitative metric reported in `main_v2.tex` corresponds directly to empirical data generated during the project and verified from disk:

| Paper Section & Table | Quantity Reported | Exact Value | Source File |
| :--- | :--- | :--- | :--- |
| **Section 3, Table 2** | `page-blocks0` stats | $N=5472, N_{\text{tr}}=4377, N_{\text{te}}=1095, d=10, n=447, \text{IR}=8.79$ | `data/splits/page-blocks0_train.csv` |
| **Section 3, Table 2** | `haberman` stats | $N=306, N_{\text{tr}}=244, N_{\text{te}}=62, d=3, n=65, \text{IR}=2.75$ | `data/splits/haberman_train.csv` |
| **Section 3, Table 2** | `sonar` stats | $N=208, N_{\text{tr}}=166, N_{\text{te}}=42, d=60, n=78, \text{IR}=1.16$ | `data/splits/sonar_train.csv` |
| **Section 3, Table 2** | `wdbc` stats | $N=569, N_{\text{tr}}=455, N_{\text{te}}=114, d=30, n=170, \text{IR}=1.68$ | `data/splits/wdbc_train.csv` |
| **Section 3, Table 2** | Global Suite Properties | 42 datasets; Median $N_{\text{tr}}=564.0$, Median $N_{\text{te}}=141.0$, Median $\text{IR}=3.53$ | `results/tables/all_datasets_comprehensive_benchmark.csv` |
| **Section 5, Table 3** | ADASYN Macro Metrics | F1: $0.6777 \pm 0.2943$, Prec: $0.6652$, Rec: $0.7031$, Spec: $0.9044$, G-Mean: $0.7726$ | `results/tables/all_datasets_macro_leaderboard.csv` |
| **Section 5, Table 3** | SMOTE-Tomek Macro Metrics | F1: $0.6705 \pm 0.2800$, Prec: $0.6679$, Rec: $0.6878$, Spec: $0.9075$, G-Mean: $0.7740$ | `results/tables/all_datasets_macro_leaderboard.csv` |
| **Section 5, Table 3** | Borderline-SMOTE-1 | F1: $0.6703 \pm 0.2990$, Prec: $0.6703$, Rec: $0.6834$, Spec: $0.9117$, G-Mean: $0.7573$ | `results/tables/all_datasets_macro_leaderboard.csv` |
| **Section 5, Table 3** | SMOTE (Standardized) | F1: $0.6617 \pm 0.2965$, Prec: $0.6644$, Rec: $0.6725$, Spec: $0.9147$, G-Mean: $0.7563$ | `results/tables/all_datasets_macro_leaderboard.csv` |
| **Section 5, Table 3** | BalancedRandomForest | F1: $0.6616 \pm 0.2697$, Prec: $0.6108$, Rec: $0.7942$, Spec: $0.8582$, G-Mean: $0.8208$ | `results/tables/all_datasets_macro_leaderboard.csv` |
| **Section 5, Table 3** | Random Oversampling | F1: $0.6535 \pm 0.3194$, Prec: $0.6858$, Rec: $0.6545$, Spec: $0.9181$, G-Mean: $0.7362$ | `results/tables/all_datasets_macro_leaderboard.csv` |
| **Section 5, Table 3** | DeepSMOTE (Autoencoder) | F1: $0.6497 \pm 0.3139$, Prec: $0.6978$, Rec: $0.6360$, Spec: $0.9353$, G-Mean: $0.7344$ | `results/tables/all_datasets_macro_leaderboard.csv` |
| **Section 5, Table 3** | RAG-Tab (BERT Semantic) | F1: $0.6254 \pm 0.3262$, Prec: $0.6959$, Rec: $0.6012$, Spec: $0.9417$, G-Mean: $0.7055$ | `results/tables/all_datasets_macro_leaderboard.csv` |
| **Section 5, Table 3** | Few-Shot LLM (Global) | F1: $0.6166 \pm 0.3302$, Prec: $0.6891$, Rec: $0.6021$, Spec: $0.9352$, G-Mean: $0.6989$ | `results/tables/all_datasets_macro_leaderboard.csv` |
| **Section 5, Table 3** | RAG-Tab (Euclidean) | F1: $0.6140 \pm 0.3334$, Prec: $0.6844$, Rec: $0.5905$, Spec: $0.9392$, G-Mean: $0.6933$ | `results/tables/all_datasets_macro_leaderboard.csv` |
| **Section 5, Table 3** | Zero-Shot LLM | F1: $0.6104 \pm 0.3293$, Prec: $0.6827$, Rec: $0.5830$, Spec: $0.9405$, G-Mean: $0.6882$ | `results/tables/all_datasets_macro_leaderboard.csv` |
| **Section 5, Table 3** | Original (No Resampling) | F1: $0.5968 \pm 0.3427$, Prec: $0.6886$, Rec: $0.5576$, Spec: $0.9519$, G-Mean: $0.6687$ | `results/tables/ablation_zero_few_rag_bert.csv` |
| **Section 5, Table 4** | Pooled Equal Budget: RAG-Tab | Prec: $0.7124$, Rec: $0.5447$, F1: $0.6003$, Spec: $0.9478$, G-Mean: $0.6886$ | `results/tables/multiclassifier_generalization_benchmark.csv` |
| **Section 5, Table 4** | Pooled Equal Budget: SMOTE | Prec: $0.7095$, Rec: $0.5437$, F1: $0.5998$, Spec: $0.9451$, G-Mean: $0.6881$ | `results/tables/multiclassifier_generalization_benchmark.csv` |
| **Section 5, Table 4** | Pooled Full SMOTE | Prec: $0.6102$, Rec: $0.6320$, F1: $0.5999$, Spec: $0.8832$, G-Mean: $0.7310$ | `results/tables/multiclassifier_generalization_benchmark.csv` |
| **Section 5, Table 5** | Random Forest: SMOTE ($N=30$) | Prec: $0.7238$, Rec: $0.5341$, F1: $0.5890$, Spec: $0.9477$, G-Mean: $0.6792$ | `results/tables/equal_budget_benchmark.csv` |
| **Section 5, Table 5** | Random Forest: RAG-Tab ($N=30$) | Prec: $0.7131$, Rec: $0.5229$, F1: $0.5779$, Spec: $0.9482$, G-Mean: $0.6675$ | `results/tables/equal_budget_benchmark.csv` |
| **Section 5, Table 6** | Cross-Classifier: HGB | RAG-Tab ($N=30$): F1 = $0.6671$ vs. Full SMOTE: F1 = $0.6305$ | `results/tables/multiclassifier_generalization_benchmark.csv` |
| **Section 5, Table 6** | Cross-Classifier: LogReg | Full SMOTE Prec: $0.5251$ vs. RAG-Tab Prec: $0.7032$ | `results/tables/multiclassifier_generalization_benchmark.csv` |
| **Section 5, Table 7 & 8** | Seed Variance (10 Datasets $\times$ 5 Seeds) | Zero/Few: Prec $0.7296 \pm 0.3353$, F1 $0.5603 \pm 0.3340$, within-seed std $\pm 0.0242$ F1 | `results/seed_check/seed_noise_benchmark.csv` |
| **Section 5, Table 9** | Distribution Fidelity & DCR | `sonar` RAG-Tab DCR: Min = $7.4064$, Mean = $8.9572$; `wdbc` Min = $4.5324$ | `results/tables/generative_fidelity_privacy_benchmark.csv` |
| **Section 7** | Precision Win Rate vs. SMOTE | Non-tied: $21/29 = 72.4\%$; Overall: $21/42 = 50.0\%$ (13 ties) | `results/tables/ablation_zero_few_rag_bert.csv` |
| **Section 7** | Wilcoxon Precision Test | One-sided: $W=306.0, p=0.02783$; Two-sided: $W=129.0, p=0.05566$ | `results/tables/ablation_zero_few_rag_bert.csv` |
| **Section 7** | Wilcoxon F1 Test vs. Original | One-sided: $W=357.0, p=0.000227$; Two-sided: $W=49.0, p=0.000454$ | `results/tables/ablation_zero_few_rag_bert.csv` |

---

## 2. Incomplete Items & TODO Markers in `main_v2.tex`

All unverified or author-dependent metadata in `main_v2.tex` are explicitly marked with `\textbf{[TODO: ...]}`:
1. **Author 1 (Corresponding):** Samhitha Reddy Madhireddy:
   - `\email{\textbf{[TODO: author.email@institution.edu]}}`
   - `\affil*[1]{\orgdiv{\textbf{[TODO: Department/Division of Computer Science]}}, \orgname{\textbf{[TODO: University/Research Institution]}}, \orgaddress{\city{\textbf{[TODO: City]}}, \country{\textbf{[TODO: Country]}}}}`
2. **Author 2:** Archana Deshmukh:
   - `\email{\textbf{[TODO: author2.email@institution.edu]}}`
   - `\affil[2]{\orgdiv{\textbf{[TODO: Department/Division of Computer Science]}}, \orgname{\textbf{[TODO: University/Research Institution]}}, \orgaddress{\city{\textbf{[TODO: City]}}, \country{\textbf{[TODO: Country]}}}}`
3. **Author 3:** Dhwani:
   - `\sur{\textbf{[TODO: Surname]}}`
   - `\email{\textbf{[TODO: author3.email@institution.edu]}}`
   - `\affil[3]{\orgdiv{\textbf{[TODO: Department/Division of Computer Science]}}, \orgname{\textbf{[TODO: University/Research Institution]}}, \orgaddress{\city{\textbf{[TODO: City]}}, \country{\textbf{[TODO: Country]}}}}`
4. **Funding:** `\textbf{[TODO: Funding information or "This research received no specific grant from any funding agency in the public, commercial, or not-for-profit sectors."]}`
5. **Author Contributions:** `\textbf{[TODO: authors to confirm contributions]}`

---

## 3. Discrepancy Reconciliation & Claims Removed from Version 1

| Original Claim / Element in `main.tex` (v1) | Empirical Finding in Audit | Action Taken in `main_v2.tex` | Justification |
| :--- | :--- | :--- | :--- |
| **Equal-Budget Caption (Table 5)** claimed: RAG-Tab outperforms equal-budget SMOTE on RandomForest ($0.7124$ Prec / $0.6003$ F1 vs. $0.7095$ Prec / $0.5998$ F1). | In `equal_budget_benchmark.csv`, on Random Forest alone, **SMOTE actually beats RAG-Tab** ($0.7238$ Prec / $0.5890$ F1 vs. $0.7131$ Prec / $0.5779$ F1). The $0.7124 / 0.6003$ numbers were the pooled average across 3 distinct classifiers (RF, HGB, LR). | **Reconciled & Split into Two Tables:** Table 4 reports the pooled 3-classifier results; Table 5 reports Random Forest alone, explicitly acknowledging SMOTE's higher precision and F1 on Random Forest. | Scientific integrity; avoids conflating pooled multi-classifier averages with single-classifier results. |
| **Generator Provenance:** Multiple sections claimed or implied that benchmark results were generated via live calls to Google Gemini Flash LLM. | In `llm_client.py`, the fallback schema-aware sampler was triggered across all committed benchmarks. The sampler samples from Gaussian distributions based on global schema ranges and does not parse tabular exemplars. | **Methodology Explicitly Discloses Fallback Mode:** Section 4.5 details the exact regex used, notes that exemplar tables are skipped, explains that Zero-Shot and Few-Shot outputs are identical, and clarifies that live Gemini API was not used for reported runs. | Prevents academic misrepresentation regarding whether an LLM was actively queried. |
| **Differential Privacy Claims:** Section heading and multiple paragraphs referred to "Differential Privacy (DCR)" and "formal privacy guarantees". | DCR (Distance to Closest Record) is a Euclidean nearest-neighbor distance diagnostic measuring minimum distance to training rows. It provides **zero mathematical $(\epsilon, \delta)$-differential privacy**. | **Removed all differential privacy wording:** Relabeled section as "Generative Fidelity and Empirical Non-Memorization (DCR)"; framed DCR strictly as a diagnostic to check against exact training data copying. | DCR cannot be equated with mathematical differential privacy. |
| **Precision Win Rate Claim:** Claimed RAG-Tab achieves a "72.4% win rate over SMOTE". | Out of 42 datasets, RAG-Tab had 21 wins, 8 losses, and **13 ties**. $21/29 = 72.4\%$, but overall win rate is $21/42 = 50.0\%$. | **Reported both denominators:** "21/29 = 72.4% non-tied AND 21/42 = 50.0% overall (13 ties)". | Omitting 13 ties inflates the perceived frequency of superiority. |
| **Statistical Significance (Wilcoxon):** Claimed "statistically significant precision superiority over SMOTE ($p = 0.0278$)". | $p = 0.0278$ is a **one-sided test**. The standard **two-sided test yields $p = 0.0557$**, which is not statistically significant at $\alpha = 0.05$. | **Explicitly labeled one-sided vs. two-sided:** Disclosed that under the standard two-sided test, precision superiority is not statistically significant ($p = 0.0557$). | Academic reporting standards require two-sided tests unless an asymmetric hypothesis was pre-registered. |
| **Nemenyi Critical Difference:** Claimed statistically significant rank separation between RAG-Tab and SMOTE on Friedman/Nemenyi tests. | The post-hoc Friedman test included Zero-Shot and Few-Shot as separate treatments even though their underlying generator produced identical samples, violating treatment independence. | **Caveated post-hoc test:** Clarified that ranking algorithms with identical treatments violates test assumptions, advising cautious interpretation. | Preserves statistical validity. |
| **Quality Control Impact:** Implied that multi-stage quality control filtered candidate samples across all datasets. | `SyntheticQualityController` was instantiated but never invoked in `run_ablation_experiments.py`; its physical rules were authored solely for `page-blocks0`. | **Transparently disclosed in Section 4.5 and Limitations:** Stated that QC filtering was bypassed during ablation benchmarks. | Accurately describes what code was executed. |

---

## 4. BibTeX Citation Audit (`references.bib`)

- **Total bib entries:** 23 entries in `references.bib`.
- **Citations in `main_v2.tex`:** 21 unique citation keys:
  `batista2004study`, `borisov2023language`, `breiman2001random`, `chawla2002smote`, `chen2016xgboost`, `dinsdale2021deepsmote`, `douzas2018improving`, `grinsztajn2022tree`, `han2005borderline`, `he2008adasyn`, `hegselmann2023tabllm`, `huang2011natural`, `ke2017lightgbm`, `kotelnikov2023tabddpm`, `le2022survey`, `mullick2020generative`, `panagiotou2024synthetic`, `reimers2019sentence`, `shwartz2022tabular`, `solatorio2023realtabformer`, `xu2019modeling`.
- **Missing from `references.bib`:** **0** (all citations in `main_v2.tex` exist and are well-formed).
- **Uncertain or hallucinated citations:** **0**. All cited works are real, peer-reviewed publications or established arXiv preprints.

---

## 5. Audit of Figures in `figures/`

The 16 image files in `figures/` were inspected without modifying the image files. Captions in `main_v2.tex` were updated to remove misleading claims:

| Figure File | Original v1 Caption / Issue | Relabeled v2 Status in `main_v2.tex` | Regeneration Recommendation |
| :--- | :--- | :--- | :--- |
| `figures/generative_fidelity_and_privacy.png` | Filename and caption claimed "Generative fidelity and empirical privacy". | Caption relabeled to: "Distribution fidelity and Distance to Closest Record (DCR) empirical memorization diagnostics across representative datasets. (Note: DCR evaluates empirical distance to training instances and does not provide formal differential privacy guarantees)." | Mark for regeneration to `figures/generative_fidelity_and_dcr.png` with "privacy" removed from the y-axis label. |
| `figures/equal_budget_comparison.png` | Caption claimed RAG-Tab outperforms equal-budget SMOTE on F1 and Precision. | Caption relabeled to: "When pooled across three classifier families, RAG-Tab and equal-budget SMOTE show comparable F1 and Precision; on Random Forest alone, equal-budget SMOTE records slightly higher precision (0.7238 vs. 0.7131) and F1 (0.5890 vs. 0.5779)." | Mark for regeneration to show both pooled and per-classifier panels. |
| `figures/critical_difference_precision.png` | Claimed statistically significant separation from SMOTE. | Caption relabeled to: "Nemenyi Critical Difference diagram for Precision across 42 benchmark datasets ($\alpha=0.05, CD=1.1635$). Note: The comparison includes Zero-Shot and Few-Shot conditions that exhibit identical underlying generator behavior." | Mark for regeneration excluding duplicate conditions. |
| `figures/critical_difference_f1.png` | Included duplicate Zero-Shot / Few-Shot conditions in ranking. | Caption relabeled with explicit caveat noting condition equivalence. | Mark for regeneration with independent treatments only. |
| `figures/ablation_zero_few_rag_bert.png` | Plotted Zero-Shot and Few-Shot as independent generative paradigms. | Caption relabeled to note that Zero-Shot and Few-Shot use identical fallback generation logic and that differences with RAG-BERT reflect sample count and top-up interpolation. | Mark for regeneration with volume-matched controls. |
| `figures/pca_tsne_manifold_diagnostics.png` | Claimed LLM-generated minority samples. | Caption relabeled to: "PCA and t-SNE manifold projections of synthetic minority samples generated by the schema-aware synthesis pipeline versus real data distributions." | Keep as distribution diagnostic. |
| `figures/neighborhood_retrieval_pca.png` | 2D PCA of Natural Neighbors on `page-blocks0`. | Caption maintained: depicts topological anchor discovery. | Retain as is. |
| `figures/multiclassifier_generalization.png` | Cross-classifier inductive bias benchmark. | Caption maintained with neutral comparison language. | Retain as is. |
| `figures/multiclass_benchmark.png` | Multiclass benchmark on 3 datasets. | Caption maintained with neutral comparison language. | Retain as is. |
| `figures/correlation_matrices_comparison.png` | Correlation matrices for `page-blocks0`. | Caption maintained with neutral comparison language. | Retain as is. |

---

## 6. Git Status and Staged Files Confirmation

- **Branch:** `audit/rag-tab-publication-readiness` (isolated from `main`).
- **`.env` Protection:** Verified that `.env` is ignored by `.gitignore` and does NOT appear in `git status`.
- **Files Staged for Local Commit:**
  1. `src/generation/llm_client.py` (strict mode, header auth, summary logging)
  2. `check_key.py` (key presence and format check without printing)
  3. `main_v2.tex` (new Springer single-column paper draft)
  4. `main_v1_backup.tex` (backup of original main.tex)
  5. `references.bib` (verified BibTeX database)
  6. `figures/` (all 16 diagnostic and benchmark figures)
  7. `results/seed_check/` (`seed_noise_benchmark.csv` and `analyze_results.py`)
  8. `WRAP_UP.md` (this comprehensive audit report)
