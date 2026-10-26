"""
SentimentScope AI — Reusable UI Components
Provides consistent headers, metric cards, sentiment badges, and layout helpers
following the monochrome SaaS design system.
"""

from typing import Optional
import streamlit as st

from src.sentimentscope.ui.styles import CUSTOM_CSS


def apply_theme():
    """Injects custom CSS theme into active Streamlit page."""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def render_header(
    title: str = "Sentiment Intelligence",
    subtitle: str = "AI-Powered Social Media Sentiment Analysis & Trend Monitoring System",
    badge_text: Optional[str] = None,
    category_label: Optional[str] = "SENTIMENTSCOPE AI",
):
    """Renders a minimalist, high-contrast SaaS page header."""
    badge_html = f"<span class='badge-neutral' style='margin-left: 10px; vertical-align: middle;'>{badge_text}</span>" if badge_text else ""
    cat_html = f"<div style='color: #8A8A8A; font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 4px;'>{category_label}</div>" if category_label else ""

    st.markdown(
        f"""
        <div class="page-header-container">
            {cat_html}
            <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 8px;">
                <h1 class="page-header-title">{title} {badge_html}</h1>
            </div>
            <p class="page-header-subtitle">{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# Alias for backward compatibility
def render_hero(
    title: str = "SentimentScope AI",
    subtitle: str = "AI-Powered Social Media Sentiment Analysis & Trend Monitoring System",
    badge_text: Optional[str] = "Production v1.0",
):
    """Backward-compatible alias for render_header."""
    render_header(title=title, subtitle=subtitle, badge_text=badge_text)


def render_metric_card(
    label: str,
    value: str,
    sub_text: str = "",
    border_accent: Optional[str] = None,
):
    """Renders a clean monochrome metric card with optional restrained accent."""
    accent_style = f"border-left: 3px solid {border_accent};" if border_accent else ""
    st.markdown(
        f"""
        <div class="metric-card" style="{accent_style}">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            {f'<div class="metric-sub">{sub_text}</div>' if sub_text else ''}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sentiment_badge(sentiment: str) -> str:
    """Returns HTML string for restrained semantic sentiment badge."""
    s = str(sentiment).strip().capitalize()
    if s == "Positive":
        return "<span class='badge-pos'>● Positive</span>"
    elif s == "Negative":
        return "<span class='badge-neg'>● Negative</span>"
    else:
        return "<span class='badge-neu'>● Neutral</span>"


def render_action_card(number: str, title: str, description: str):
    """Renders a clean action card for workflow navigation."""
    st.markdown(
        f"""
        <div class="action-card">
            <div style="font-size: 11px; font-weight: 600; color: #8A8A8A; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 4px;">Action {number}</div>
            <h4>{title}</h4>
            <p>{description}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
