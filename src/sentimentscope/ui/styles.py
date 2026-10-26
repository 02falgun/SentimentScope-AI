"""
SentimentScope AI — Custom CSS Styles and Monochrome Design System Tokens
Implements a minimalist, high-contrast black-and-white SaaS aesthetic
with restrained semantic colors for sentiment classification.
"""

CUSTOM_CSS = """
<style>
/* Import Inter Font */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* Global Font & Text Reset */
html, body, .stApp {
    font-family: 'Inter', system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    color: #111111;
}

/* Explicitly preserve Streamlit Material Icons and Glyphs */
[data-testid="stIconMaterial"],
.material-symbols-rounded,
.material-symbols-outlined,
.material-icons,
span[class*="material-symbols"],
i[class*="material-icons"],
span[data-testid="stIconMaterial"] {
    font-family: 'Material Symbols Rounded', 'Material Symbols Outlined', 'Material Icons' !important;
    font-weight: normal !important;
    font-style: normal !important;
    text-transform: none !important;
    letter-spacing: normal !important;
    word-wrap: normal !important;
    white-space: nowrap !important;
    direction: ltr !important;
    display: inline-block !important;
    line-height: 1 !important;
}

/* Ensure Expander and Sidebar collapse icon containers don't leak ligatures as text */
button[data-testid="stSidebarCollapseButton"] span[data-testid="stIconMaterial"],
details[data-testid="stExpander"] summary span[data-testid="stIconMaterial"] {
    font-family: 'Material Symbols Rounded', 'Material Symbols Outlined', 'Material Icons' !important;
}

/* Primary Application Canvas */
.stApp {
    background-color: #FFFFFF !important;
    color: #111111 !important;
}

/* Clean Header Bar */
header[data-testid="stHeader"] {
    background-color: rgba(255, 255, 255, 0.95) !important;
    border-bottom: 1px solid #E5E5E5 !important;
}

/* Sidebar Shell */
section[data-testid="stSidebar"] {
    background-color: #FAFAFA !important;
    border-right: 1px solid #E5E5E5 !important;
    padding-top: 1.5rem !important;
}

section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
    color: #444444 !important;
}

/* Sidebar Radio Navigation */
section[data-testid="stSidebar"] div[role="radiogroup"] > label {
    background: transparent;
    padding: 8px 12px;
    margin-bottom: 4px;
    border-radius: 6px;
    border: 1px solid transparent;
    transition: all 0.15s ease;
    cursor: pointer;
}

section[data-testid="stSidebar"] div[role="radiogroup"] > label:hover {
    background: #F0F0F0;
    border-color: #E5E5E5;
}

section[data-testid="stSidebar"] div[role="radiogroup"] > label[data-checked="true"],
section[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) {
    background: #FFFFFF !important;
    border: 1px solid #E5E5E5 !important;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
    font-weight: 600;
}

/* Modern Page Header Container (Non-Gradient, Monochrome SaaS) */
.page-header-container {
    background-color: #FFFFFF;
    border: 1px solid #E5E5E5;
    border-radius: 8px;
    padding: 24px 28px;
    margin-bottom: 24px;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.03);
}

.page-header-title {
    font-size: 24px;
    font-weight: 700;
    color: #111111;
    letter-spacing: -0.02em;
    margin: 0;
    display: flex;
    align-items: center;
    gap: 12px;
}

.page-header-subtitle {
    color: #666666;
    font-size: 14px;
    margin-top: 6px;
    margin-bottom: 0;
    line-height: 1.5;
}

/* Monochrome Metric Cards */
.metric-card {
    background-color: #FFFFFF;
    border: 1px solid #E5E5E5;
    border-radius: 8px;
    padding: 18px 20px;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.02);
    transition: border-color 0.15s ease, box-shadow 0.15s ease;
    height: 100%;
}

.metric-card:hover {
    border-color: #D4D4D4;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.04);
}

.metric-label {
    color: #666666;
    font-size: 12px;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 4px;
}

.metric-value {
    color: #111111;
    font-size: 26px;
    font-weight: 700;
    font-variant-numeric: tabular-nums;
    line-height: 1.2;
}

.metric-sub {
    color: #8A8A8A;
    font-size: 12px;
    margin-top: 4px;
    line-height: 1.4;
}

/* Restrained Semantic Sentiment Badges */
.badge-pos {
    background-color: #F0FDF4;
    color: #15803D;
    border: 1px solid #BBF7D0;
    padding: 3px 10px;
    border-radius: 9999px;
    font-size: 12px;
    font-weight: 600;
    display: inline-flex;
    align-items: center;
    gap: 4px;
}

.badge-neu {
    background-color: #F8FAFC;
    color: #475569;
    border: 1px solid #E2E8F0;
    padding: 3px 10px;
    border-radius: 9999px;
    font-size: 12px;
    font-weight: 600;
    display: inline-flex;
    align-items: center;
    gap: 4px;
}

.badge-neg {
    background-color: #FEF2F2;
    color: #B91C1C;
    border: 1px solid #FECACA;
    padding: 3px 10px;
    border-radius: 9999px;
    font-size: 12px;
    font-weight: 600;
    display: inline-flex;
    align-items: center;
    gap: 4px;
}

.badge-neutral {
    background-color: #F5F5F5;
    color: #444444;
    border: 1px solid #E5E5E5;
    padding: 3px 10px;
    border-radius: 9999px;
    font-size: 12px;
    font-weight: 500;
    display: inline-flex;
    align-items: center;
}

.badge-warning {
    background-color: #FFFBEB;
    color: #B45309;
    border: 1px solid #FDE68A;
    padding: 6px 12px;
    border-radius: 6px;
    font-size: 13px;
    font-weight: 500;
}

/* Token Attribution Tags */
.token-support {
    background-color: #F0FDF4;
    color: #15803D;
    border: 1px solid #BBF7D0;
    padding: 3px 8px;
    border-radius: 4px;
    margin: 3px 4px 3px 0;
    display: inline-block;
    font-size: 13px;
    font-weight: 500;
}

.token-oppose {
    background-color: #FEF2F2;
    color: #B91C1C;
    border: 1px solid #FECACA;
    padding: 3px 8px;
    border-radius: 4px;
    margin: 3px 4px 3px 0;
    display: inline-block;
    font-size: 13px;
    font-weight: 500;
}

.token-neutral {
    background-color: #F5F5F5;
    color: #666666;
    border: 1px solid #E5E5E5;
    padding: 3px 8px;
    border-radius: 4px;
    margin: 3px 4px 3px 0;
    display: inline-block;
    font-size: 13px;
}

/* Status Cards in Sidebar */
.sidebar-status-card {
    background-color: #FFFFFF;
    border: 1px solid #E5E5E5;
    border-radius: 6px;
    padding: 10px 12px;
    margin-top: 6px;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.02);
}

.status-dot-active {
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background-color: #15803D;
    margin-right: 6px;
}

.status-dot-inactive {
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background-color: #B91C1C;
    margin-right: 6px;
}

.status-dot-neutral {
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background-color: #64748B;
    margin-right: 6px;
}

/* Primary Action Buttons (High Contrast Black with Pure White Text) */
.stButton > button[kind="primary"],
.stButton > button[data-testid="baseButton-primary"],
div[data-testid="stFormSubmitButton"] > button,
div[data-testid="stFormSubmitButton"] > button[data-testid="baseButton-primary"],
.stDownloadButton > button,
.stDownloadButton > button[data-testid="baseButton-primary"] {
    background-color: #000000 !important;
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
    border: 1px solid #000000 !important;
    border-radius: 6px !important;
    font-weight: 500 !important;
    font-size: 13px !important;
    transition: all 0.15s ease !important;
}

.stButton > button[kind="primary"] *,
.stButton > button[data-testid="baseButton-primary"] *,
div[data-testid="stFormSubmitButton"] > button *,
div[data-testid="stFormSubmitButton"] > button[data-testid="baseButton-primary"] *,
.stDownloadButton > button *,
.stDownloadButton > button[data-testid="baseButton-primary"] * {
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
}

.stButton > button[kind="primary"]:hover,
.stButton > button[data-testid="baseButton-primary"]:hover,
div[data-testid="stFormSubmitButton"] > button:hover,
.stDownloadButton > button:hover {
    background-color: #262626 !important;
    border-color: #262626 !important;
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
}

.stButton > button[kind="primary"]:active,
.stButton > button[data-testid="baseButton-primary"]:active,
div[data-testid="stFormSubmitButton"] > button:active,
.stDownloadButton > button:active {
    background-color: #404040 !important;
    border-color: #404040 !important;
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
}

.stButton > button[kind="primary"]:focus-visible,
.stButton > button[data-testid="baseButton-primary"]:focus-visible,
div[data-testid="stFormSubmitButton"] > button:focus-visible,
.stDownloadButton > button:focus-visible {
    outline: 2px solid #000000 !important;
    outline-offset: 2px !important;
}

.stButton > button[kind="primary"]:disabled,
.stButton > button[data-testid="baseButton-primary"]:disabled,
div[data-testid="stFormSubmitButton"] > button:disabled,
.stDownloadButton > button:disabled {
    background-color: #737373 !important;
    border-color: #737373 !important;
    color: #D4D4D4 !important;
    -webkit-text-fill-color: #D4D4D4 !important;
    cursor: not-allowed !important;
}

/* Secondary / Default Action Buttons (Pure White with Dark Text) */
.stButton > button[kind="secondary"],
.stButton > button[data-testid="baseButton-secondary"],
.stButton > button:not([kind="primary"]):not([data-testid="baseButton-primary"]) {
    background-color: #FFFFFF !important;
    color: #111111 !important;
    -webkit-text-fill-color: #111111 !important;
    border: 1px solid #D4D4D4 !important;
    border-radius: 6px !important;
    font-weight: 500 !important;
    font-size: 13px !important;
    transition: all 0.15s ease !important;
}

.stButton > button[kind="secondary"] *,
.stButton > button[data-testid="baseButton-secondary"] *,
.stButton > button:not([kind="primary"]):not([data-testid="baseButton-primary"]) * {
    color: #111111 !important;
    -webkit-text-fill-color: #111111 !important;
}

.stButton > button[kind="secondary"]:hover,
.stButton > button[data-testid="baseButton-secondary"]:hover,
.stButton > button:not([kind="primary"]):not([data-testid="baseButton-primary"]):hover {
    background-color: #F5F5F5 !important;
    border-color: #111111 !important;
    color: #000000 !important;
    -webkit-text-fill-color: #000000 !important;
}

.stButton > button[kind="secondary"]:active,
.stButton > button[data-testid="baseButton-secondary"]:active,
.stButton > button:not([kind="primary"]):not([data-testid="baseButton-primary"]):active {
    background-color: #E5E5E5 !important;
    border-color: #000000 !important;
    color: #000000 !important;
    -webkit-text-fill-color: #000000 !important;
}

.stButton > button[kind="secondary"]:focus-visible,
.stButton > button[data-testid="baseButton-secondary"]:focus-visible,
.stButton > button:not([kind="primary"]):not([data-testid="baseButton-primary"]):focus-visible {
    outline: 2px solid #111111 !important;
    outline-offset: 2px !important;
}

.stButton > button[kind="secondary"]:disabled,
.stButton > button[data-testid="baseButton-secondary"]:disabled,
.stButton > button:not([kind="primary"]):not([data-testid="baseButton-primary"]):disabled {
    background-color: #FAFAFA !important;
    border-color: #E5E5E5 !important;
    color: #A3A3A3 !important;
    -webkit-text-fill-color: #A3A3A3 !important;
    cursor: not-allowed !important;
}

/* Expander Clean Monochrome Styling */
details[data-testid="stExpander"] {
    background-color: #FFFFFF !important;
    border: 1px solid #E5E5E5 !important;
    border-radius: 8px !important;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.02) !important;
}

details[data-testid="stExpander"] summary {
    font-weight: 600 !important;
    color: #111111 !important;
    padding: 10px 14px !important;
}

details[data-testid="stExpander"] summary:hover {
    color: #000000 !important;
}

/* Callout / Info Containers */
.info-callout {
    background-color: #F8FAFC;
    border: 1px solid #E2E8F0;
    border-left: 4px solid #111111;
    border-radius: 6px;
    padding: 14px 18px;
    margin: 14px 0;
    color: #334155;
    font-size: 13px;
    line-height: 1.5;
}

/* Quick Action Cards */
.action-card {
    background-color: #FFFFFF;
    border: 1px solid #E5E5E5;
    border-radius: 8px;
    padding: 18px 20px;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.02);
    transition: all 0.15s ease;
    height: 100%;
}

.action-card:hover {
    border-color: #111111;
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.05);
}

.action-card h4 {
    margin: 0 0 6px 0;
    color: #111111;
    font-size: 15px;
    font-weight: 600;
}

.action-card p {
    color: #666666;
    font-size: 13px;
    line-height: 1.45;
    margin: 0;
}

/* Academic Callout Banner */
.academic-callout {
    background-color: #FAFAFA;
    border: 1px solid #E5E5E5;
    border-radius: 8px;
    padding: 20px 24px;
    margin: 16px 0;
}

.academic-callout h4 {
    color: #111111;
    font-size: 16px;
    font-weight: 600;
    margin-top: 0;
    margin-bottom: 8px;
}

/* Text Inputs and Selectboxes */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea,
.stSelectbox > div > div {
    background-color: #FFFFFF !important;
    border: 1px solid #E5E5E5 !important;
    border-radius: 6px !important;
    color: #111111 !important;
    font-size: 14px !important;
}

.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {
    border-color: #111111 !important;
    box-shadow: 0 0 0 1px #111111 !important;
}

/* Streamlit Tabs */
.stTabs [data-baseweb="tab-list"] {
    border-bottom: 1px solid #E5E5E5;
    gap: 8px;
}

.stTabs [data-baseweb="tab"] {
    background-color: transparent;
    border: none;
    color: #666666;
    font-weight: 500;
    padding: 8px 16px;
}

.stTabs [aria-selected="true"] {
    color: #111111 !important;
    font-weight: 600 !important;
    border-bottom: 2px solid #000000 !important;
}
</style>
"""
