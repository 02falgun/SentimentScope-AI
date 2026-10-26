# SentimentScope AI — Security Policy & Ephemeral Data Handling

## 1. Uploaded Data Privacy Policy

In accordance with PRD Section 2.5 (Non-Functional Requirements: Security):
- **Ephemeral Session Processing:** Datasets uploaded by users via the Streamlit interface are loaded **strictly into in-memory session memory** (`st.session_state`).
- **Zero Permanent Storage:** The application does not save, write, or cache user-uploaded CSV files to the server's local filesystem or any external database.
- **Session Expiration:** When the user closes or refreshes their browser tab, all uploaded records, filtered views, and session prediction histories are garbage-collected and destroyed.

---

## 2. Model Serialization Security

- Model artifacts are saved using `joblib` with schema validation in `model_registry.py`.
- On loading, the system validates the existence and types of expected dictionary keys (`pipeline`, `model_name`, `classes`) and confirms that the pipeline is an instance of `sklearn.pipeline.Pipeline`.
- **Untrusted Artifact Guard:** Never load `.joblib` files from untrusted third-party sources. Only load models produced by `scripts/train_model.py` or the in-app ML Lab.

---

## 3. Secrets & API Credential Safety

- **No Secrets in Source:** The repository contains zero API keys, database passwords, or machine-specific tokens.
- **`.gitignore` Enforcement:** The `.gitignore` file explicitly excludes `.env`, `secrets.toml`, virtual environments (`.venv/`), and OS temporary files.
