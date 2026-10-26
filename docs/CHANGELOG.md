# SentimentScope AI — Changelog

All notable changes to this project are documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [2.0.0] - 2026-10-26 — Premium UI/UX Transformation & Monochrome SaaS Design System

### Added
- **Monochrome SaaS Design System:**
  - Implemented crisp white canvas (`#FFFFFF`) with dark typography (`#111111`), subtle borders (`#E5E5E5`), and restrained semantic sentiment markers.
  - Authored comprehensive specification in `docs/DESIGN_SYSTEM.md` and UI/UX audit in `docs/UI_UX_AUDIT.md`.
  - Re-themed `.streamlit/config.toml` to light monochrome mode (`primaryColor="#000000"`, `backgroundColor="#FFFFFF"`, `secondaryBackgroundColor="#F5F5F5"`).
  - Authored modular CSS rules in `src/sentimentscope/ui/styles.py` replacing dark glassmorphism and neon gradients with clean SaaS layout cards, tabular numerals, and Inter typography.
- **Application Shell & Navigation Redesign:**
  - Redesigned `app.py` sidebar with high-contrast `SentimentScope AI` branding, clean section radio pills, and unified Model Active and Dataset Active status cards.
  - Reusable page header (`render_header`) with breadcrumb categories and metadata badges in `src/sentimentscope/ui/components.py`.
  - Upgraded metric cards (`render_metric_card`) with tabular numerals, muted uppercase labels, and restrained left-border sentiment accents.
- **Core Workspace Overhauls (All 8 Pages):**
  - **Overview Dashboard (`home.py`):** "Sentiment Intelligence" executive layout, active corpus summary, 4 primary KPI cards, Plotly donut and volume bar charts on clean white canvas, model status banner, and structured quick action tiles.
  - **Live Prediction (`prediction_page.py`):** Dedicated AI analysis workspace, quick test case chips, 500-char input validation, black primary action button, outcome card with predicted sentiment, calibrated confidence score, inference latency, probability progress bars, and token-level feature attribution tags (`.token-support`, `.token-oppose`).
  - **ML Lab & Benchmark (`ml_lab.py`):** Cross-algorithm leaderboard, high-visibility Scientific Audit expander on Group-CV template isolation, confusion matrix with clean sequential slate heatmap, ROC curves, and misclassified sample inspector.
  - **Dataset Explorer (`dataset_explorer.py`):** In-memory ingestion panel, structured column schema mapping, row attrition quality metrics, and responsive preview table.
  - **Visual Analytics (`analytics_page.py`):** Light theme (`LIGHT_LAYOUT`) across all 8 visualizations, grayscale engagement and character length bars, restrained sentiment colors, and white-canvas Matplotlib WordCloud.
  - **Keyword Search (`search_page.py`):** Prominent search input, polarity sentiment index card, donut chart, and matching records table.
  - **Reports & Export (`reports_page.py`):** Executive summary preview cards and high-contrast download buttons for PDF and CSV.
  - **About & Architecture (`about_page.py`):** 3-tier system architecture diagram, ML methodology cards, and updated viva examination guide.
- **Reporting & Visualizations:**
  - Refactored `src/sentimentscope/analytics/visualizations.py` to `LIGHT_LAYOUT` with subtle `#F0F0F0` gridlines and restrained palette (`#15803D` Positive, `#64748B` Neutral, `#B91C1C` Negative).
  - Overhauled `src/sentimentscope/reporting/pdf_report.py` to monochrome executive format with pure black rules, Helvetica typography, and clean table borders.
- **Visual Verification & QA:**
  - Automated visual verification across all 8 pages executed via headless browser subagent on `http://localhost:8505`.
  - All 29 unit tests passing (`pytest tests/ -v`).
  - Model smoke tests passing (`python scripts/validate_model.py`).
  - Code hygiene verified with `git diff --check`.

---

## [1.0.1] - 2026-10-26 — Post-Deployment Audit & Scientific Generalization Release

### Added
- **Scientific 100% Accuracy Investigation:**
  - Implemented `evaluate_template_generalization` in `src/sentimentscope/ml/evaluation.py` to calculate Group K-Fold Cross-Validation isolating unseen base sentence templates.
  - Documented empirical finding: 93.1% base sentence template overlap in standard 80/20 train/test splits explains in-template memorization.
  - Validated out-of-template generalization benchmark: 93.15% Accuracy and 0.9193 Macro F1 across 5 folds.
  - Evaluated zero-shot generalization on 11 real-world social media posts, achieving 90.9% accuracy with calibrated uncertainty intervals.
  - Added interactive scientific audit expander to ML Lab (`src/sentimentscope/ui/pages/ml_lab.py`).
  - Added viva demonstration talking points on template leakage to About page (`src/sentimentscope/ui/pages/about_page.py`).
- **Defensive Engineering & Robustness:**
  - Added safe character capping (`len(text) <= 2000`) in `predict_single_text` (`src/sentimentscope/ml/prediction.py`) to prevent latency/memory spikes on pathological inputs.
  - Added safe empty-dataset handling to `plot_sentiment_donut` and `plot_sentiment_bars` in `src/sentimentscope/analytics/visualizations.py`.
  - Added Ground-Truth vs Model-Prediction distinction and toggling to Home page (`src/sentimentscope/ui/pages/home.py`).
- **Automated QA Expansion:**
  - Added `test_plot_sentiment_empty_data` in `tests/test_analytics.py`.
  - Added `test_evaluate_template_generalization` in `tests/test_training.py`.
  - Added `test_predict_single_text_excessive_length` in `tests/test_prediction.py`.
  - Expanded test suite to **29/29 passing tests** (100% pass rate).

---

## [1.0.0] - 2026-10-26 — Initial Production Release

### Added
- Baseline architecture, scikit-learn models, 8 initial pages, dark theme, and automated test suite.
