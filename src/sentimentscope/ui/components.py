"""
SentimentScope AI — Reusable UI Components
Renders hero banners, metric cards, badges, and layout helpers.
"""

from typing import Optional
import streamlit as st

from src.sentimentscope.ui.styles import CUSTOM_CSS


def apply_theme():
    """Injects custom CSS theme into active Streamlit page."""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def render_hero(
    title: str = "SentimentScope AI",
    subtitle: str = "AI-Powered Social Media Sentiment Analysis & Trend Monitoring System",
    badge_text: Optional[str] = "ML Lab Project v1.0",
):
    """Renders the top hero banner."""
    badge_html = f"<span class='badge-pos' style='margin-left: 12px; vertical-align: middle;'>{badge_text}</span>" if badge_text else ""
    st.markdown(
        f"""
        <div class="hero-container">
            <div style="display: flex; align-items: center; justify-content: space-between;">
                <div>
                    <h1 class="hero-title">{title} {badge_html}</h1>
                    <p class="hero-subtitle">{subtitle}</p>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_card(
    label: str,
    value: str,
    sub_text: str = "",
    border_accent: Optional[str] = None,
):
    """Renders a glassmorphism metric card."""
    style_attr = f"style='border-left: 4px solid {border_accent};'" if border_accent else ""
    st.markdown(
        f"""
        <div class="metric-card" {style_attr}>
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            {f'<div class="metric-sub">{sub_text}</div>' if sub_text else ''}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sentiment_badge(sentiment: str) -> str:
    """Returns HTML string for colored sentiment badge."""
    s = str(sentiment).capitalize()
    if s == "Positive":
        return f"<span class='badge-pos'>● Positive</span>"
    elif s == "Negative":
        return f"<span class='badge-neg'>● Negative</span>"
    else:
        return f"<span class='badge-neu'>● Neutral</span>"
