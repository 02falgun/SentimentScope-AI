"""
SentimentScope AI — Custom CSS Styles and Theme Tokens
Implements sleek dark mode, glassmorphism, badges, and responsive typography.
"""

CUSTOM_CSS = """
<style>
/* Import Inter Font */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
}

/* App Background & Header */
.stApp {
    background: radial-gradient(circle at 15% 15%, rgba(99, 102, 241, 0.08) 0%, transparent 40%),
                radial-gradient(circle at 85% 85%, rgba(16, 185, 129, 0.05) 0%, transparent 40%),
                #0F172A;
    color: #F8FAFC;
}

/* Hero Banner */
.hero-container {
    background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.8) 100%);
    border: 1px solid rgba(255, 255, 255, 0.08);
    backdrop-filter: blur(12px);
    border-radius: 16px;
    padding: 24px 32px;
    margin-bottom: 24px;
    box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
}

.hero-title {
    font-size: 28px;
    font-weight: 700;
    background: linear-gradient(90deg, #FFFFFF 0%, #C7D2FE 50%, #818CF8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
}

.hero-subtitle {
    color: #94A3B8;
    font-size: 14px;
    margin-top: 6px;
    margin-bottom: 0px;
}

/* Glassmorphism Metric Cards */
.metric-card {
    background: rgba(30, 41, 59, 0.6);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 12px;
    padding: 16px 20px;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
    transition: transform 0.2s ease, border-color 0.2s ease;
}

.metric-card:hover {
    border-color: rgba(99, 102, 241, 0.4);
    transform: translateY(-2px);
}

.metric-label {
    color: #94A3B8;
    font-size: 12px;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.metric-value {
    color: #F8FAFC;
    font-size: 26px;
    font-weight: 700;
    margin-top: 4px;
}

.metric-sub {
    color: #64748B;
    font-size: 11px;
    margin-top: 2px;
}

/* Badges */
.badge-pos {
    background: rgba(16, 185, 129, 0.15);
    color: #10B981;
    border: 1px solid rgba(16, 185, 129, 0.3);
    padding: 4px 10px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 600;
    display: inline-block;
}

.badge-neu {
    background: rgba(100, 116, 139, 0.15);
    color: #94A3B8;
    border: 1px solid rgba(100, 116, 139, 0.3);
    padding: 4px 10px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 600;
    display: inline-block;
}

.badge-neg {
    background: rgba(239, 68, 68, 0.15);
    color: #EF4444;
    border: 1px solid rgba(239, 68, 68, 0.3);
    padding: 4px 10px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 600;
    display: inline-block;
}

.badge-warning {
    background: rgba(245, 158, 11, 0.15);
    color: #F59E0B;
    border: 1px solid rgba(245, 158, 11, 0.3);
    padding: 4px 12px;
    border-radius: 6px;
    font-size: 13px;
    font-weight: 500;
}

/* Token Attribution Tags */
.token-support {
    background: rgba(16, 185, 129, 0.2);
    color: #34D399;
    border-bottom: 2px solid #10B981;
    padding: 2px 6px;
    border-radius: 4px;
    margin: 2px;
    display: inline-block;
    font-size: 13px;
}

.token-oppose {
    background: rgba(239, 68, 68, 0.2);
    color: #F87171;
    border-bottom: 2px solid #EF4444;
    padding: 2px 6px;
    border-radius: 4px;
    margin: 2px;
    display: inline-block;
    font-size: 13px;
}

.token-neutral {
    background: rgba(100, 116, 139, 0.15);
    color: #94A3B8;
    padding: 2px 6px;
    border-radius: 4px;
    margin: 2px;
    display: inline-block;
    font-size: 13px;
}

/* Sidebar Styling */
section[data-testid="stSidebar"] {
    background-color: #0B1120;
    border-right: 1px solid rgba(255, 255, 255, 0.05);
}

/* Buttons */
.stButton>button {
    border-radius: 8px;
    font-weight: 500;
    transition: all 0.2s ease;
}

.stButton>button:hover {
    border-color: #6366F1;
    box-shadow: 0 0 12px rgba(99, 102, 241, 0.3);
}
</style>
"""
