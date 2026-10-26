# SentimentScope AI — Architectural Decision Records (ADR)

## ADR-001: Classical Machine Learning vs. Deep Learning / Transformers

- **Status:** Accepted
- **Context:** PRD Section 1.10 specifies classical machine learning models (Logistic Regression, Naive Bayes, Linear SVM, Random Forest) for the baseline release, deferring LLMs and deep learning models to future phases.
- **Decision:** Use scikit-learn pipelines with TF-IDF vectorization.
- **Consequences:**
  - Fast training (~1.2 seconds total).
  - Sub-millisecond prediction latency (0.04 ms/sample).
  - Low memory footprint (<180 MB RAM), ideal for free-tier Streamlit Community Cloud hosting.
  - Direct linear interpretability via model coefficients.

---

## ADR-002: Negation Preservation in Text Preprocessing

- **Status:** Accepted
- **Context:** Standard English stopwords lists strip words such as `not`, `never`, `no`, `hardly`, `without`. In sentiment analysis, removing these words causes false classifications (e.g. *"not bad"* becoming *"bad"*, *"never buy again"* becoming *"buy"*).
- **Decision:** Define `NEGATION_TOKENS` and exclude them from `EFFECTIVE_STOPWORDS`.
- **Consequences:** Negations are preserved in tokens and n-grams, enabling accurate classification of nuanced phrasing.

---

## ADR-003: Linear SVM Probability Calibration

- **Status:** Accepted
- **Context:** Linear Support Vector Machines (`LinearSVC`) optimize hinge loss and output hyperplane decision scores, not calibrated probabilities.
- **Decision:** Wrap `LinearSVC` with `CalibratedClassifierCV(estimator=base_svc, cv=3)` during pipeline construction.
- **Consequences:** Calibration is fitted strictly on training folds without test leakage, allowing SVM to produce valid probability distributions.

---

## ADR-004: In-Memory Upload Lifecycle (No File Persistence)

- **Status:** Accepted
- **Context:** The PRD requires that user-uploaded CSVs must not be persisted beyond the active session.
- **Decision:** Parse uploaded files into `st.session_state.active_df` and do not write files to disk.
- **Consequences:** Satisfies data privacy requirements and avoids disk quota issues on Streamlit Cloud.
