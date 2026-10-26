# SentimentScope AI — Troubleshooting & FAQ

## 1. Installation & Environment Issues

### Issue: `pip: command not found`
- **Cause:** Using a raw system Python binary where pip is not symlinked in PATH.
- **Solution:** Invoke pip via python module syntax:
  ```bash
  python3 -m pip install -r requirements.txt
  ```
  Or create a virtual environment with Python 3.11 or 3.12:
  ```bash
  python3.12 -m venv .venv
  source .venv/bin/activate
  pip install -r requirements.txt
  ```

---

## 2. Model Artifact Issues

### Issue: `Model artifact not found at models/sentiment_model.joblib`
- **Cause:** First-time clone without trained artifacts.
- **Solution:** Execute the standalone training script:
  ```bash
  python scripts/train_model.py
  ```
  Alternatively, launching `streamlit run app.py` will automatically train a baseline model on the built-in sample dataset on first launch.

---

## 3. CSV Ingestion & Encoding Errors

### Issue: `UnicodeDecodeError: 'utf-8' codec can't decode byte...`
- **Cause:** Dataset saved in Windows CP1252 or Latin-1 encoding.
- **Solution:** SentimentScope AI's `load_csv` automatically tries `utf-8`, `utf-8-sig`, `latin1`, `iso-8859-1`, and `cp1252`. If an upload fails, save the file as standard UTF-8 in Excel or LibreOffice and re-upload.

### Issue: `Required text column 'X' not found in dataset`
- **Cause:** Uploaded CSV uses non-standard header names (e.g. `user_comment`, `tweet_body`).
- **Solution:** In **Dataset Explorer**, use the **Column Schema Mapping** dropdown to manually designate the text column.

---

## 4. PDF Generation Warnings

### Issue: `Character outside range supported by font`
- **Cause:** Using non-ASCII unicode characters with built-in Helvetica fonts in FPDF.
- **Solution:** The `safe_text()` utility automatically sanitizes special punctuation (em-dashes, smart quotes) into ASCII equivalents before rendering the PDF.
