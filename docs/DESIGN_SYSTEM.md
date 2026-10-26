# SentimentScope AI — Monochrome SaaS Design System Specification

**Version:** 2.0 (Production Release)  
**Philosophy:** Monochrome interface + meaningful data colors + exceptional layout quality.  
**Design Intent:** A minimalist, high-density AI analytics dashboard inspired by modern enterprise SaaS products (Linear, Vercel, Stripe), engineered for academic credibility, viva examinations, and production deployments.

---

## 1. Core Color System

The interface is strictly dominated by black, white, and neutral grays. Colors are never used for arbitrary decoration. Color is strictly reserved for semantic sentiment state communication and analytical matrix heatmaps.

### 1.1 Neutral Palette (Interface Canvas & Containers)

| Design Token | Hex Code | Role in Application |
| :--- | :--- | :--- |
| `primary-background` | `#FFFFFF` | Primary viewport background and chart plot background. |
| `secondary-surface` | `#FAFAFA` / `#F5F5F5` | Sidebar background, secondary container fills, table header fills. |
| `subtle-hover` | `#F0F0F0` | Interactive card hovers, radio button hovers, dropdown item hover. |
| `primary-border` | `#E5E5E5` | Structural 1px separation lines, card outlines, table borders. |
| `hover-border` | `#D4D4D4` | Elevated hover state for metric cards and interactive tiles. |
| `main-text` | `#111111` | Primary headings, KPI figures, table cell data, button labels. |
| `secondary-text` | `#666666` | Form labels, card descriptions, metric titles, axis labels. |
| `muted-text` | `#8A8A8A` | Captions, timestamps, secondary units, inactive breadcrumbs. |
| `primary-button-bg` | `#000000` | High-emphasis call to actions ("Analyze Sentiment", "Run ML Benchmark"). |
| `primary-button-text` | `#FFFFFF` | Primary button text label. |
| `primary-button-hover` | `#262626` | Dark charcoal hover state for primary action buttons. |

### 1.2 Restrained Semantic Color Palette (Data & Sentiment Only)

Sentiment colors are carefully tuned for contrast against `#FFFFFF` and `#FAFAFA` surfaces:

| Sentiment Class | Accent Hex | Surface Tint | Border Hex | Usage |
| :--- | :--- | :--- | :--- | :--- |
| **Positive** | `#15803D` | `#F0FDF4` | `#BBF7D0` | Favorable sentiment badge, positive chart bars, positive tokens. |
| **Neutral** | `#64748B` | `#F8FAFC` | `#E2E8F0` | Objective/neutral badge, neutral chart bars, neutral tokens. |
| **Negative** | `#B91C1C` | `#FEF2F2` | `#FECACA` | Critical/negative badge, negative chart bars, negative tokens. |
| **Uncertainty / Warning** | `#B45309` | `#FFFBEB` | `#FDE68A` | Low-confidence model warning, data quality alerts. |

> **Accessibility Rule:** Sentiment is never conveyed via color alone. Every badge, chart bar, and prediction result pairs color with explicit text labels, icon markers (`●`), and percentages.

---

## 2. Typography & Hierarchical Scale

The typography uses `Inter` (with native system-sans fallback: Apple System, Segoe UI, Roboto) loaded via Google Fonts.

| Hierarchy Element | Font Size | Weight | Line Height | Tracking | Color |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **App Title (Sidebar)** | 18px | 700 | 1.2 | -0.02em | `#111111` |
| **Page Title (`h1`)** | 24px | 700 | 1.25 | -0.02em | `#111111` |
| **Section Heading (`h2` / `h3`)** | 18px | 600 | 1.3 | -0.01em | `#111111` |
| **Subsection Heading (`h4`)** | 15px | 600 | 1.4 | normal | `#111111` |
| **Metric Value (KPI)** | 26px | 700 | 1.2 | normal (tabular) | `#111111` |
| **Metric Label** | 12px | 500 | 1.0 | +0.05em (caps) | `#666666` |
| **Body Text** | 14px | 400 | 1.5 | normal | `#333333` |
| **Secondary & Captions** | 12px | 400 | 1.4 | normal | `#8A8A8A` |
| **Badge Text** | 12px | 600 | 1.0 | normal | Contextual |

### Tabular Numerals
All metric cards, classification reports, confusion matrix cells, and latency values enforce `font-variant-numeric: tabular-nums;` to prevent layout jitter during dynamic updates.

---

## 3. Layout, Spacing & Container Rules

1. **Max Content Width & Margins:**
   - Wide layout enabled via `st.set_page_config(layout="wide")`.
   - Consistent container margins (padding: 24px 28px).
2. **Card Restraint:**
   - Elements are **not** arbitrarily wrapped in nested boxes. Cards are strictly reserved for:
     - Top-level KPI summaries (`.metric-card`).
     - Major interactive analysis blocks (Live prediction input and outcome).
     - Next-step quick actions (`.action-card`).
3. **Corner Radii & Borders:**
   - Containers & metric cards: `border-radius: 8px; border: 1px solid #E5E5E5;`
   - Buttons & form controls: `border-radius: 6px;`
   - Sentiment badges: `border-radius: 9999px;` (Pill format).
   - Shadows: Subtle `0 1px 2px rgba(0, 0, 0, 0.03)` only. No heavy drop-shadows or neon glow effects.

---

## 4. Reusable UI Components

### 4.1 Page Header (`render_header`)
```python
render_header(
    title="Sentiment Intelligence",
    subtitle="AI-Powered Social Media Sentiment Analysis & Trend Monitoring System",
    badge_text="Production v1.0",
    category_label="SENTIMENTSCOPE AI"
)
```
Renders a high-contrast SaaS header card with breadcrumb category, clean title, optional metadata badge, and explanatory subtitle.

### 4.2 KPI Metric Card (`render_metric_card`)
```python
render_metric_card(
    label="Active Dataset",
    value="360",
    sub_text="Cleaned posts ready for inference",
    border_accent="#15803D" # Optional subtle left border
)
```

### 4.3 Sentiment Badge (`render_sentiment_badge`)
```python
render_sentiment_badge("Positive")  # <span class='badge-pos'>● Positive</span>
render_sentiment_badge("Neutral")   # <span class='badge-neu'>● Neutral</span>
render_sentiment_badge("Negative")  # <span class='badge-neg'>● Negative</span>
```

### 4.4 Token Attribution Tags
- Supporting Positive/Class Token: `.token-support` (Soft green background `#F0FDF4`, border `#BBF7D0`, text `#15803D`).
- Opposing/Negative Token: `.token-oppose` (Soft red background `#FEF2F2`, border `#FECACA`, text `#B91C1C`).
- Neutral Token: `.token-neutral` (Subtle gray background `#F5F5F5`, border `#E5E5E5`, text `#666666`).

---

## 5. Chart Visualization Rules (`Plotly` & `Matplotlib`)

All Plotly charts inherit the light monochrome layout:
```python
LIGHT_LAYOUT = dict(
    paper_bgcolor="#FFFFFF",
    plot_bgcolor="#FFFFFF",
    font=dict(color="#111111", family="Inter, system-ui, sans-serif"),
    margin=dict(l=20, r=20, t=40, b=20),
)
```

1. **Donut & Bar Charts:** White slice separators (`marker=dict(line=dict(color="#FFFFFF", width=2))`), restrained green/gray/red colors, legible bold percentages.
2. **Time Series & Line Charts:** White canvas with subtle `#F0F0F0` horizontal gridlines. Minimal marker sizes (size=5), distinct line colors for Positive (`#15803D`), Neutral (`#64748B`), Negative (`#B91C1C`).
3. **Confusion Matrix:** High-contrast sequential palette (`Greys` or `Blues` scale from white to dark navy) with bold numerical contrast annotations.
4. **ROC Curves:** Crisp one-vs-rest lines with dashed gray reference diagonal (`#94A3B8`).
5. **WordCloud:** High-contrast dark words on a pure `#FFFFFF` background.
