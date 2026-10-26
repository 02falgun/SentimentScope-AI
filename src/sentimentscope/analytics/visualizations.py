"""
SentimentScope AI — Interactive Visualizations Module
Renders responsive Plotly charts and WordClouds with consistent theme tokens.
"""

from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.sentimentscope.config import CANONICAL_SENTIMENTS, SENTIMENT_COLOR_MAP

# Dark Theme Tokens
DARK_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#F8FAFC", family="Inter, system-ui, sans-serif"),
    margin=dict(l=20, r=20, t=40, b=20),
)


def plot_sentiment_donut(distribution: Dict[str, int], title: str = "Sentiment Share") -> go.Figure:
    """Renders a polished donut chart of sentiment distribution."""
    labels = [cls for cls in CANONICAL_SENTIMENTS if distribution.get(cls, 0) > 0]
    values = [distribution.get(cls, 0) for cls in labels]
    colors = [SENTIMENT_COLOR_MAP.get(cls, "#6366F1") for cls in labels]

    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=values,
                hole=0.55,
                marker=dict(colors=colors, line=dict(color="#0F172A", width=2)),
                textinfo="label+percent",
                hoverinfo="label+value+percent",
                insidetextfont=dict(color="#FFFFFF", size=13),
            )
        ]
    )
    fig.update_layout(
        **DARK_LAYOUT,
        title=dict(text=title, x=0.5, font=dict(size=16, color="#F8FAFC")),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
    )
    return fig


def plot_sentiment_bars(distribution: Dict[str, int], title: str = "Sentiment Volume") -> go.Figure:
    """Renders a high-contrast bar chart of sentiment counts."""
    classes = CANONICAL_SENTIMENTS
    counts = [distribution.get(cls, 0) for cls in classes]
    colors = [SENTIMENT_COLOR_MAP[cls] for cls in classes]

    fig = go.Figure(
        data=[
            go.Bar(
                x=classes,
                y=counts,
                marker=dict(color=colors, line=dict(color="#0F172A", width=1.5)),
                text=counts,
                textposition="auto",
                hoverinfo="x+y",
            )
        ]
    )
    fig.update_layout(
        **DARK_LAYOUT,
        title=dict(text=title, x=0.5, font=dict(size=16, color="#F8FAFC")),
        xaxis=dict(title="Sentiment Category", gridcolor="#334155"),
        yaxis=dict(title="Total Posts", gridcolor="#334155"),
    )
    return fig


def plot_daily_trends(trends_df: pd.DataFrame) -> go.Figure:
    """Renders a multi-line time series chart for daily sentiment trajectories."""
    if trends_df.empty:
        fig = go.Figure()
        fig.update_layout(**DARK_LAYOUT, title="No date data available")
        return fig

    fig = go.Figure()
    for cls in CANONICAL_SENTIMENTS:
        if cls in trends_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=trends_df["Date"],
                    y=trends_df[cls],
                    name=cls,
                    mode="lines+markers",
                    line=dict(color=SENTIMENT_COLOR_MAP[cls], width=2.5),
                    marker=dict(size=6),
                )
            )

    fig.update_layout(
        **DARK_LAYOUT,
        title=dict(text="Daily Sentiment Volume Trend", x=0.5, font=dict(size=16)),
        xaxis=dict(title="Date", gridcolor="#334155"),
        yaxis=dict(title="Post Count", gridcolor="#334155"),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
    )
    return fig


def plot_platform_comparison(platform_df: pd.DataFrame) -> go.Figure:
    """Renders a grouped bar chart comparing sentiment across platforms."""
    if platform_df.empty:
        fig = go.Figure()
        fig.update_layout(**DARK_LAYOUT, title="No platform data available")
        return fig

    fig = go.Figure()
    for cls in CANONICAL_SENTIMENTS:
        if cls in platform_df.columns:
            fig.add_trace(
                go.Bar(
                    name=cls,
                    x=platform_df["Platform"],
                    y=platform_df[cls],
                    marker=dict(color=SENTIMENT_COLOR_MAP[cls]),
                )
            )

    fig.update_layout(
        **DARK_LAYOUT,
        barmode="group",
        title=dict(text="Cross-Platform Sentiment Comparison", x=0.5, font=dict(size=16)),
        xaxis=dict(title="Social Media Platform", gridcolor="#334155"),
        yaxis=dict(title="Post Volume", gridcolor="#334155"),
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
    )
    return fig


def plot_engagement_comparison(engagement_df: pd.DataFrame) -> go.Figure:
    """Renders average likes and shares grouped by sentiment."""
    if engagement_df.empty or "Avg_Likes" not in engagement_df.columns:
        fig = go.Figure()
        fig.update_layout(**DARK_LAYOUT, title="No engagement metrics available")
        return fig

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name="Avg Likes",
            x=engagement_df["Sentiment"],
            y=engagement_df["Avg_Likes"],
            marker=dict(color="#38BDF8"),
        )
    )
    if "Avg_Shares" in engagement_df.columns:
        fig.add_trace(
            go.Bar(
                name="Avg Shares",
                x=engagement_df["Sentiment"],
                y=engagement_df["Avg_Shares"],
                marker=dict(color="#A855F7"),
            )
        )

    fig.update_layout(
        **DARK_LAYOUT,
        barmode="group",
        title=dict(text="Average Engagement by Sentiment Class", x=0.5, font=dict(size=16)),
        xaxis=dict(title="Sentiment Class", gridcolor="#334155"),
        yaxis=dict(title="Average Count", gridcolor="#334155"),
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
    )
    return fig


def plot_post_length_histogram(char_lengths: List[int]) -> go.Figure:
    """Renders histogram of post character lengths."""
    fig = go.Figure(
        data=[
            go.Histogram(
                x=char_lengths,
                nbinsx=25,
                marker=dict(color="#818CF8", line=dict(color="#0F172A", width=1)),
            )
        ]
    )
    fig.update_layout(
        **DARK_LAYOUT,
        title=dict(text="Post Character Length Distribution", x=0.5, font=dict(size=16)),
        xaxis=dict(title="Character Length", gridcolor="#334155"),
        yaxis=dict(title="Post Count", gridcolor="#334155"),
    )
    return fig


def plot_top_words_bar(
    word_freqs: List[Tuple[str, int]],
    title: str = "Top Associated Terms",
    bar_color: str = "#6366F1",
) -> go.Figure:
    """Renders a horizontal bar chart of top terms."""
    if not word_freqs:
        fig = go.Figure()
        fig.update_layout(**DARK_LAYOUT, title="No words found")
        return fig

    words = [wf[0] for wf in word_freqs][::-1]
    counts = [wf[1] for wf in word_freqs][::-1]

    fig = go.Figure(
        data=[
            go.Bar(
                x=counts,
                y=words,
                orientation="h",
                marker=dict(color=bar_color),
            )
        ]
    )
    fig.update_layout(
        **DARK_LAYOUT,
        title=dict(text=title, x=0.5, font=dict(size=15)),
        xaxis=dict(title="Frequency", gridcolor="#334155"),
        yaxis=dict(title="Word", gridcolor="#334155"),
    )
    return fig


def plot_confusion_matrix(cm: List[List[int]], classes: List[str]) -> go.Figure:
    """Renders an annotated confusion matrix heatmap."""
    fig = go.Figure(
        data=go.Heatmap(
            z=cm,
            x=[f"Pred: {c}" for c in classes],
            y=[f"True: {c}" for c in classes],
            colorscale="Viridis",
            text=cm,
            texttemplate="%{text}",
            textfont=dict(size=14, color="#FFFFFF"),
        )
    )
    fig.update_layout(
        **DARK_LAYOUT,
        title=dict(text="Confusion Matrix (Held-Out Test Set)", x=0.5, font=dict(size=16)),
        xaxis=dict(title="Predicted Sentiment Class", gridcolor="#334155"),
        yaxis=dict(title="True Ground-Truth Class", gridcolor="#334155", autorange="reversed"),
    )
    return fig


def plot_roc_curves(roc_curves: Optional[Dict[str, Dict[str, List[float]]]]) -> go.Figure:
    """Renders multi-class ROC curves with diagonal reference line."""
    fig = go.Figure()
    fig.add_shape(
        type="line",
        line=dict(dash="dash", color="#64748B"),
        x0=0, x1=1, y0=0, y1=1,
    )

    if roc_curves:
        for cls, curves in roc_curves.items():
            color = SENTIMENT_COLOR_MAP.get(cls, "#6366F1")
            fig.add_trace(
                go.Scatter(
                    x=curves["fpr"],
                    y=curves["tpr"],
                    name=f"ROC: {cls}",
                    mode="lines",
                    line=dict(color=color, width=2.5),
                )
            )

    fig.update_layout(
        **DARK_LAYOUT,
        title=dict(text="One-vs-Rest ROC Curves", x=0.5, font=dict(size=16)),
        xaxis=dict(title="False Positive Rate", gridcolor="#334155", range=[0, 1]),
        yaxis=dict(title="True Positive Rate", gridcolor="#334155", range=[0, 1.05]),
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
    )
    return fig


def generate_wordcloud_figure(text_corpus: str, bg_color: str = "#1E293B") -> Optional[plt.Figure]:
    """Generates a Matplotlib WordCloud figure if wordcloud library is available."""
    try:
        from wordcloud import WordCloud
        if not text_corpus.strip():
            return None

        wc = WordCloud(
            width=800,
            height=400,
            background_color=bg_color,
            colormap="cool",
            stopwords=EFFECTIVE_STOPWORDS,
            max_words=100,
        ).generate(text_corpus)

        fig, ax = plt.subplots(figsize=(8, 4), facecolor=bg_color)
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        plt.tight_layout(pad=0)
        return fig
    except Exception:
        return None
