# SentimentScope AI — UI/UX Audit & Transformation Assessment

**Date:** October 2026  
**Auditor:** Senior AI Analytics UI/UX & Machine Learning Systems Architect  
**Project:** SentimentScope AI — AI-Powered Social Media Sentiment Analysis & Trend Monitoring System  
**Audit Objective:** Comprehensive structural, visual, and ergonomic audit of the existing Streamlit web interface prior to the Monochrome SaaS Design System overhaul.

---

## 1. Executive Summary & Audit Context

SentimentScope AI is a functional, academically rigorous machine learning analytics application deployed on Streamlit Community Cloud. While the underlying data pipeline, scikit-learn models, validation routines, and reporting services are fully operational (29/29 passing tests, sub-millisecond latency, leakage-safe Group-CV generalization benchmarks), the interface reflects the aesthetic and organizational limitations of early-stage prototyping:

1. **Aesthetic Misalignment:** The application currently employs a generic dark slate theme (`#0F172A`) with high-saturation neon purple/indigo accents (`#6366F1`), gradient text titles, and glowing card borders. This design detracts from the credibility of an AI analytics and research platform.
2. **Chart Styling Disconnect:** Data visualizations in Plotly and Matplotlib utilize dark transparent backgrounds with mixed neon colors (cyan `#38BDF8`, purple `#A855F7`, amber `#F59E0B`), rather than a disciplined palette of monochrome containers and restrained semantic sentiment markers.
3. **Container Overuse & Inconsistent Visual Hierarchy:** Every element is wrapped in heavy, bordered cards (`.metric-card`), creating visual fatigue and clutter. Header titles and badges vary in alignment and typography across the 8 sections.
4. **Navigation Ergonomics:** The standard Streamlit sidebar radio control has basic labels with emoji prefixes and lacks refined active-state indicators and cohesive branding.
5. **Technical Constraints:** The application runs on Streamlit Community Cloud (Python 3.12, Linux container, resource-constrained free tier). All visual improvements must be achieved using pure Streamlit widgets, responsive custom CSS injection, and Plotly chart configuration, without introducing heavy front-end node build steps or breaking Cloud deployment compatibility.

---

## 2. Section-by-Section Usability & Visual Audit

### 2.1 Navigation & Application Shell (`app.py`, `styles.py`, `components.py`)
- **Current State:** Dark sidebar (`#0B1120`) with glowing border gradients; system title has mixed emoji and text; active page selection relies on default radio circle buttons; model and dataset status indicators use colored backgrounds with glowing borders.
- **Defects / Deficiencies:**
  - High saturation colors distract from primary analytical content.
  - Sidebar lacks clean SaaS hierarchy (branding, page list, system health, version stamp).
  - Main container has excessive radial gradients (`radial-gradient(circle at 15% 15%...)`).
- **Planned Improvements:**
  - Clean monochrome shell with `#FFFFFF` primary background and subtle `#FAFAFA` sidebar surface.
  - Minimalist brandmark: `SentimentScope AI` with clean typography (`Inter`, 600 weight, 18px) and subtitle `AI Sentiment & Trend Intelligence`.
  - Refined status badges using neutral light cards (`#F5F5F5` surface, `#E5E5E5` border) with subtle semantic status dots (`#15803D` for active model).
  - Streamlit radio navigation styled with clean hover and active pill states.

### 2.2 Home / Overview (`ui/pages/home.py`)
- **Current State:** Hero banner with gradient text (`hero-title` linear-gradient `#FFFFFF` to `#818CF8`); four metric cards with colored left borders; two Plotly charts (donut and bar); info banner for active model; three quick-start cards.
- **Defects / Deficiencies:**
  - The hero gradient text looks like a prototype template rather than a serious research tool.
  - Colored borders on every metric card create visual clutter.
  - Chart backgrounds are dark with high-contrast colored legends.
- **Planned Improvements:**
  - Redesign as "Sentiment Intelligence Dashboard" with crisp typography, breadcrumbs, and active dataset context.
  - Clean KPI cards using tabular numerals, subtle `#E5E5E5` borders, muted labels (`#666666`), and high-contrast metric values (`#111111`).
  - Restrained semantic colors: Positive (`#15803D`), Neutral (`#64748B`), Negative (`#B91C1C`).
  - Cohesive layout for distribution charts (donut and volume) on a clean white canvas.

### 2.3 Live Prediction (`ui/pages/prediction_page.py`)
- **Current State:** Sample prompt buttons in 5 columns; text area for user input; analyze button with blue styling; prediction result cards with glowing accents; probability progress bars; token attribution tags with green/red backgrounds.
- **Defects / Deficiencies:**
  - Buttons for sample prompts wrap awkwardly on smaller viewports.
  - Result metrics (predicted sentiment, confidence, latency) are spaced unevenly.
  - Token attribution highlights are overly vibrant (`rgba(16, 185, 129, 0.2)` and `rgba(239, 68, 68, 0.2)`).
- **Planned Improvements:**
  - Dedicated AI inference workspace: clean side-by-side or stacked grid with crisp prompt chips.
  - Character counter with defensive validation (max 500 chars).
  - Solid black primary action button (`#000000`, white text, `#262626` hover).
  - Polished outcome card: prominent sentiment badge, calibrated confidence score, and inference latency.
  - Clean probability bars with monochrome tracks and semantic fills.
  - Refined token-level explainability tags with subtle background tints (`#F0FDF4` and `#FEF2F2`) and 1px borders.

### 2.4 Dataset Explorer (`ui/pages/dataset_explorer.py`)
- **Current State:** File uploader; reset button; 3-column dropdowns for column mapping; 4 data quality cards; batch prediction trigger; table filters and dataframe view.
- **Defects / Deficiencies:**
  - Column mapping dropdowns look cluttered and lack visual grouping.
  - Data quality cards use contrasting neon accents.
  - Table preview is constrained by default Streamlit theme colors.
- **Planned Improvements:**
  - Clean file ingestion panel with clear drag-and-drop boundary and in-memory security notice.
  - Structured schema mapping section with distinct column groupings (Mandatory Text vs Optional Metadata).
  - Attrition metrics displayed as a cohesive tabular or flow summary.
  - Full-width responsive table preview with clean search and platform filters.

### 2.5 ML Lab (`ui/pages/ml_lab.py`)
- **Current State:** Leaderboard table; Group-CV explanation expander; 4 metric cards; confusion matrix (Viridis heatmap); ROC curve plot; per-class report table; error inspector; retrain controls.
- **Defects / Deficiencies:**
  - Viridis heatmap for the confusion matrix clashes with the monochrome aesthetic.
  - The Group-CV audit is one of the project's strongest academic findings, but is hidden inside a standard expander with unstyled text.
  - The retrain section looks unfinished.
- **Planned Improvements:**
  - Academic-grade benchmarking suite with a clean leaderboard table highlighting the production-selected model.
  - High-visibility Scientific Audit callout highlighting the Group-CV out-of-distribution benchmark (93.15% accuracy, 0.9193 Macro F1).
  - High-contrast confusion matrix heatmap using a clean sequential monochrome or slate palette with clear numerical annotations.
  - Clean multi-class ROC curves with distinct stroke patterns and semantic colors.
  - Interactive misclassification inspector with side-by-side comparison of ground-truth vs predicted labels.

### 2.6 Visual Analytics (`ui/pages/analytics_page.py`)
- **Current State:** Filter expander; distribution charts; daily trend multi-line chart; platform grouped bar chart; engagement bar chart; character length histogram; top words horizontal bars; Matplotlib WordCloud; trending hashtags and emojis.
- **Defects / Deficiencies:**
  - Too many competing colors across 8 different charts.
  - WordCloud uses dark blue background (`#1E293B`) with `cool` colormap, creating visual inconsistency with other charts.
  - Daily trend line chart uses thin colored lines that lack clarity on high-DPI displays.
- **Planned Improvements:**
  - Unified chart theme (`LIGHT_LAYOUT`) with white background, subtle `#F0F0F0` gridlines, and bold typography.
  - Grayscale bar charts for non-sentiment variables (e.g., character length histogram, engagement counts).
  - Semantic colors strictly reserved for Positive (`#15803D`), Neutral (`#64748B`), and Negative (`#B91C1C`).
  - WordCloud rendered on a clean white canvas with high-contrast dark text.

### 2.7 Keyword Search (`ui/pages/search_page.py`)
- **Current State:** Search input box; search button; 3 metric cards; donut chart; matching posts table.
- **Defects / Deficiencies:**
  - Search input and action button lack alignment.
  - Empty search state does not provide guidance on suggested queries.
- **Planned Improvements:**
  - Premium search bar with instant query suggestions (e.g., `performance`, `update`, `bug`, `battery`, `UI`).
  - Polarity sentiment index metric card with high-contrast indicators.
  - Clean matching records table with keyword context highlighting.

### 2.8 Reports & Export (`ui/pages/reports_page.py` & `pdf_report.py`)
- **Current State:** Executive preview cards; PDF download button; CSV download button; PDF generation script uses purple lines (`#6366F1`) and dark slate headers.
- **Defects / Deficiencies:**
  - PDF styling does not reflect the monochrome brand identity.
  - Export cards feel disconnected from the summary preview.
- **Planned Improvements:**
  - Clean executive summary panel with dataset provenance and model validation metrics.
  - Re-styled PDF document: pure black-and-white header rules, crisp Helvetica typography, grayscale table headers, and clean page numbering.
  - High-contrast primary download actions for PDF and CSV.

### 2.9 About & Architecture (`ui/pages/about_page.py`)
- **Current State:** ASCII architecture diagram inside a code block; methodology cards; faculty viva talking points.
- **Defects / Deficiencies:**
  - ASCII art is useful but raw; lacks modern technical presentation.
  - Dense text without structured visual hierarchy.
- **Planned Improvements:**
  - Polished technical architecture view with clean multi-tier pipeline explanation (Ingestion -> Preprocessing -> Benchmark/Serving -> Analytics).
  - Clear viva demonstration cheat sheet highlighting key examiner questions (Why classical ML? How does explainability work? Why was 100% test accuracy audited with Group-CV?).

---

## 3. Design Tokens & Color Strategy

| Token | Previous Value | Redesign Token | Role & Rationale |
| :--- | :--- | :--- | :--- |
| **Primary Background** | `#0F172A` (Dark slate) | `#FFFFFF` | Crisp, modern canvas matching premier AI SaaS tools. |
| **Secondary Surface** | `#1E293B` (Blue-gray) | `#F5F5F5` / `#FAFAFA` | Subtle differentiation for cards, sidebars, and tables. |
| **Main Text** | `#F8FAFC` (Off-white) | `#111111` | Maximum contrast, readable typography. |
| **Secondary Text** | `#94A3B8` (Slate gray) | `#666666` | Clear hierarchy for labels and metadata. |
| **Muted Text** | `#64748B` (Darker slate) | `#8A8A8A` | Captions, timestamps, and secondary info. |
| **Primary Border** | `rgba(255,255,255,0.08)` | `#E5E5E5` | Clean, subtle 1px structural boundaries. |
| **Primary Button** | `#6366F1` (Indigo neon) | `#000000` | Confident, minimalist black button with `#FFFFFF` text. |
| **Button Hover** | `#4F46E5` | `#262626` | Refined dark charcoal hover state. |
| **Positive Sentiment** | `#10B981` (Emerald) | `#15803D` | Restrained forest/emerald green with high accessibility. |
| **Neutral Sentiment** | `#64748B` (Slate) | `#64748B` | Balanced neutral slate. |
| **Negative Sentiment** | `#EF4444` (Crimson) | `#B91C1C` | Restrained deep red with high accessibility. |

---

## 4. Action Plan & Implementation Sequence

1. **Phase 1 — Design System Foundation:**
   - Update `.streamlit/config.toml` to enforce light mode with black/white core tokens.
   - Refactor `styles.py` and `components.py` with the complete monochrome design system and reusable components.
   - Update `config.py` sentiment color mappings to restrained tones.
   - Author `docs/DESIGN_SYSTEM.md`.
2. **Phase 2 — Navigation & Application Shell:**
   - Redesign `app.py` sidebar with professional branding, monochrome status cards, and clean typography.
3. **Phase 3 — Page Redesigns (Overview, Prediction, ML Lab, Explorer, Analytics, Search, Reports, About):**
   - Systematically refactor all 8 pages to adhere strictly to the shared design system.
4. **Phase 4 — Chart & PDF Overhaul:**
   - Refactor `visualizations.py` to light layout (`LIGHT_LAYOUT`) with monochrome structure and restrained sentiment colors.
   - Refactor `pdf_report.py` to clean monochrome executive styling.
5. **Phase 5 — Testing & Visual Verification:**
   - Execute test suite (`.venv/bin/pytest`) and validation script (`scripts/validate_model.py`).
   - Visually verify rendered pages using browser inspection.
6. **Phase 6 — Final Documentation:**
   - Update `CHANGELOG.md`, `IMPLEMENTATION_PLAN.md`, `TESTING.md`, `DEPLOYMENT.md`, and `FINAL_HANDOVER.md`.
