"""
SentimentScope AI — Interactive Visualizations Module
Renders responsive Plotly charts and WordClouds with consistent monochrome
design system tokens and restrained semantic colors.
"""

from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.sentimentscope.config import CANONICAL_SENTIMENTS, SENTIMENT_COLOR_MAP
from src.sentimentscope.data.preprocessing import EFFECTIVE_STOPWORDS

# Monochrome SaaS Light Theme Tokens
LIGHT_LAYOUT = dict(
    paper_bgcolor="#FFFFFF",
    plot_bgcolor="#FFFFFF",
    font=dict(color="#111111", family="Inter, system-ui, -apple-system, sans-serif"),
)
DEFAULT_MARGIN = dict(l=40, r=30, t=50, b=40)

AXIS_STYLE = dict(
    showgrid=True,
    gridcolor="#F0F0F0",
    gridwidth=1,
    zeroline=True,
    zerolinecolor="#E5E5E5",
    linecolor="#E5E5E5",
    tickfont=dict(color="#666666", size=11),
    title_font=dict(color="#111111", size=12, family="Inter, sans-serif"),
)


def plot_sentiment_donut(distribution: Dict[str, int], title: str = "Sentiment Share") -> go.Figure:
    """Renders a polished donut chart of sentiment distribution."""
    labels = [cls for cls in CANONICAL_SENTIMENTS if distribution.get(cls, 0) > 0]
    values = [distribution.get(cls, 0) for cls in labels]
    colors = [SENTIMENT_COLOR_MAP.get(cls, "#64748B") for cls in labels]

    if not labels or sum(values) == 0:
        fig = go.Figure()
        fig.update_layout(
            **LIGHT_LAYOUT,
            margin=DEFAULT_MARGIN,
            title=dict(text=f"{title} (No Data)", x=0.5, font=dict(size=14, color="#666666")),
            annotations=[dict(text="No sentiment records available", x=0.5, y=0.5, font_size=13, showarrow=False, font=dict(color="#8A8A8A"))],
        )
        return fig

    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=values,
                hole=0.58,
                marker=dict(colors=colors, line=dict(color="#FFFFFF", width=2.5)),
                textinfo="label+percent",
                hoverinfo="label+value+percent",
                textfont=dict(size=12, color="#FFFFFF"),
            )
        ]
    )
    fig.update_layout(
        **LIGHT_LAYOUT,
        margin=dict(l=30, r=30, t=50, b=55),
        title=dict(text=title, x=0.5, font=dict(size=15, color="#111111", weight=600)),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.22,
            xanchor="center",
            x=0.5,
            font=dict(color="#666666", size=11),
        ),
    )
    return fig


def plot_sentiment_bars(distribution: Dict[str, int], title: str = "Sentiment Volume") -> go.Figure:
    """Renders a clean bar chart of sentiment counts using restrained semantic colors."""
    classes = CANONICAL_SENTIMENTS
    counts = [distribution.get(cls, 0) for cls in classes]
    colors = [SENTIMENT_COLOR_MAP[cls] for cls in classes]

    if sum(counts) == 0:
        fig = go.Figure()
        fig.update_layout(
            **LIGHT_LAYOUT,
            margin=DEFAULT_MARGIN,
            title=dict(text=f"{title} (No Data)", x=0.5, font=dict(size=14, color="#666666")),
            annotations=[dict(text="No sentiment records available", x=0.5, y=0.5, font_size=13, showarrow=False, font=dict(color="#8A8A8A"))],
        )
        return fig

    fig = go.Figure(
        data=[
            go.Bar(
                x=classes,
                y=counts,
                marker=dict(color=colors, line=dict(color="#E5E5E5", width=1)),
                text=counts,
                textposition="auto",
                textfont=dict(color="#FFFFFF", size=12, weight=600),
                hoverinfo="x+y",
            )
        ]
    )
    fig.update_layout(
        **LIGHT_LAYOUT,
        margin=dict(l=60, r=30, t=50, b=55),
        title=dict(text=title, x=0.5, font=dict(size=15, color="#111111", weight=600)),
        xaxis=dict(**AXIS_STYLE, title="Sentiment Category"),
        yaxis=dict(**AXIS_STYLE, title="Total Posts"),
    )
    return fig


def plot_daily_trends(trends_df: pd.DataFrame) -> go.Figure:
    """Renders a multi-line time series chart for daily sentiment trajectories."""
    if trends_df.empty:
        fig = go.Figure()
        fig.update_layout(**LIGHT_LAYOUT, margin=DEFAULT_MARGIN, title="No date data available")
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
                    line=dict(color=SENTIMENT_COLOR_MAP[cls], width=2),
                    marker=dict(size=5, line=dict(color="#FFFFFF", width=1)),
                )
            )

    fig.update_layout(
        **LIGHT_LAYOUT,
        margin=dict(l=60, r=30, t=50, b=75),
        title=dict(text="Daily Sentiment Volume Trend", x=0.5, font=dict(size=15, color="#111111", weight=600)),
        xaxis=dict(**AXIS_STYLE, title="Date"),
        yaxis=dict(**AXIS_STYLE, title="Post Count"),
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.28,
            xanchor="center",
            x=0.5,
            font=dict(color="#666666", size=11),
        ),
    )
    return fig


def plot_platform_comparison(platform_df: pd.DataFrame) -> go.Figure:
    """Renders a grouped bar chart comparing sentiment across platforms."""
    if platform_df.empty:
        fig = go.Figure()
        fig.update_layout(**LIGHT_LAYOUT, margin=DEFAULT_MARGIN, title="No platform data available")
        return fig

    fig = go.Figure()
    for cls in CANONICAL_SENTIMENTS:
        if cls in platform_df.columns:
            fig.add_trace(
                go.Bar(
                    name=cls,
                    x=platform_df["Platform"],
                    y=platform_df[cls],
                    marker=dict(color=SENTIMENT_COLOR_MAP[cls], line=dict(color="#FFFFFF", width=1)),
                )
            )

    fig.update_layout(
        **LIGHT_LAYOUT,
        margin=dict(l=60, r=30, t=50, b=75),
        barmode="group",
        title=dict(text="Cross-Platform Sentiment Comparison", x=0.5, font=dict(size=15, color="#111111", weight=600)),
        xaxis=dict(**AXIS_STYLE, title="Social Media Platform"),
        yaxis=dict(**AXIS_STYLE, title="Post Volume"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.28,
            xanchor="center",
            x=0.5,
            font=dict(color="#666666", size=11),
        ),
    )
    return fig


def plot_engagement_comparison(engagement_df: pd.DataFrame) -> go.Figure:
    """Renders average likes and shares grouped by sentiment using disciplined grayscale."""
    if engagement_df.empty or "Avg_Likes" not in engagement_df.columns:
        fig = go.Figure()
        fig.update_layout(**LIGHT_LAYOUT, margin=DEFAULT_MARGIN, title="No engagement metrics available")
        return fig

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name="Avg Likes",
            x=engagement_df["Sentiment"],
            y=engagement_df["Avg_Likes"],
            marker=dict(color="#111111", line=dict(color="#FFFFFF", width=1)),
        )
    )
    if "Avg_Shares" in engagement_df.columns:
        fig.add_trace(
            go.Bar(
                name="Avg Shares",
                x=engagement_df["Sentiment"],
                y=engagement_df["Avg_Shares"],
                marker=dict(color="#64748B", line=dict(color="#FFFFFF", width=1)),
            )
        )

    fig.update_layout(
        **LIGHT_LAYOUT,
        margin=dict(l=60, r=30, t=50, b=75),
        barmode="group",
        title=dict(text="Average Engagement by Sentiment Class", x=0.5, font=dict(size=15, color="#111111", weight=600)),
        xaxis=dict(**AXIS_STYLE, title="Sentiment Class"),
        yaxis=dict(**AXIS_STYLE, title="Average Count"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.28,
            xanchor="center",
            x=0.5,
            font=dict(color="#666666", size=11),
        ),
    )
    return fig


def plot_post_length_histogram(char_lengths: List[int]) -> go.Figure:
    """Renders histogram of post character lengths using restrained monochrome tone."""
    fig = go.Figure(
        data=[
            go.Histogram(
                x=char_lengths,
                nbinsx=25,
                marker=dict(color="#334155", line=dict(color="#FFFFFF", width=1)),
            )
        ]
    )
    fig.update_layout(
        **LIGHT_LAYOUT,
        margin=dict(l=60, r=30, t=50, b=55),
        title=dict(text="Post Character Length Distribution", x=0.5, font=dict(size=15, color="#111111", weight=600)),
        xaxis=dict(**AXIS_STYLE, title="Character Length"),
        yaxis=dict(**AXIS_STYLE, title="Post Count"),
    )
    return fig


def plot_top_words_bar(
    word_freqs: List[Tuple[str, int]],
    title: str = "Top Associated Terms",
    bar_color: str = "#111111",
) -> go.Figure:
    """Renders a horizontal bar chart of top terms."""
    if not word_freqs:
        fig = go.Figure()
        fig.update_layout(**LIGHT_LAYOUT, margin=DEFAULT_MARGIN, title="No words found")
        return fig

    words = [wf[0] for wf in word_freqs][::-1]
    counts = [wf[1] for wf in word_freqs][::-1]

    fig = go.Figure(
        data=[
            go.Bar(
                x=counts,
                y=words,
                orientation="h",
                marker=dict(color=bar_color, line=dict(color="#FFFFFF", width=1)),
            )
        ]
    )
    fig.update_layout(
        **LIGHT_LAYOUT,
        margin=dict(l=85, r=30, t=50, b=55),
        title=dict(text=title, x=0.5, font=dict(size=14, color="#111111", weight=600)),
        xaxis=dict(**AXIS_STYLE, title="Frequency"),
        yaxis=dict(**AXIS_STYLE, title="Term"),
    )
    return fig


def plot_confusion_matrix(cm: List[List[int]], classes: List[str]) -> go.Figure:
    """Renders an annotated confusion matrix heatmap using a clean sequential slate palette."""
    colorscale = [
        [0.0, "#F8FAFC"],
        [0.5, "#94A3B8"],
        [1.0, "#0F172A"],
    ]

    fig = go.Figure(
        data=go.Heatmap(
            z=cm,
            x=[f"Pred: {c}" for c in classes],
            y=[f"True: {c}" for c in classes],
            colorscale=colorscale,
            text=cm,
            texttemplate="%{text}",
            textfont=dict(size=13, family="Inter, sans-serif"),
            showscale=True,
            colorbar=dict(
                tickfont=dict(color="#666666", size=10),
                thickness=12,
                len=0.8,
            ),
        )
    )
    fig.update_layout(
        **LIGHT_LAYOUT,
        margin=dict(l=85, r=40, t=55, b=65),
        title=dict(text="Confusion Matrix (Held-Out Test Set)", x=0.5, font=dict(size=15, color="#111111", weight=600)),
        xaxis=dict(**AXIS_STYLE, title="Predicted Sentiment Class"),
        yaxis=dict(**AXIS_STYLE, title="True Ground-Truth Class", autorange="reversed"),
    )
    return fig


def plot_roc_curves(roc_curves: Optional[Dict[str, Dict[str, List[float]]]]) -> go.Figure:
    """Renders multi-class ROC curves with diagonal reference line on a clean canvas."""
    fig = go.Figure()
    fig.add_shape(
        type="line",
        line=dict(dash="dash", color="#CBD5E1", width=1.5),
        x0=0, x1=1, y0=0, y1=1,
    )

    if roc_curves:
        for cls, curves in roc_curves.items():
            color = SENTIMENT_COLOR_MAP.get(cls, "#64748B")
            fig.add_trace(
                go.Scatter(
                    x=curves["fpr"],
                    y=curves["tpr"],
                    name=f"ROC: {cls}",
                    mode="lines",
                    line=dict(color=color, width=2),
                )
            )

    fig.update_layout(
        **LIGHT_LAYOUT,
        margin=dict(l=65, r=35, t=55, b=75),
        title=dict(text="One-vs-Rest ROC Curves", x=0.5, font=dict(size=15, color="#111111", weight=600)),
        xaxis=dict(**AXIS_STYLE, title="False Positive Rate", range=[0, 1]),
        yaxis=dict(**AXIS_STYLE, title="True Positive Rate", range=[0, 1.05]),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.28,
            xanchor="center",
            x=0.5,
            font=dict(color="#666666", size=11),
        ),
    )
    return fig


def generate_wordcloud_figure(text_corpus: str, bg_color: str = "#FFFFFF") -> Optional[plt.Figure]:
    """Generates a clean Matplotlib WordCloud figure on a pure white canvas."""
    try:
        from wordcloud import WordCloud
        if not text_corpus.strip():
            return None

        wc = WordCloud(
            width=800,
            height=380,
            background_color=bg_color,
            colormap="bone",
            stopwords=EFFECTIVE_STOPWORDS,
            max_words=100,
            prefer_horizontal=0.9,
        ).generate(text_corpus)

        fig, ax = plt.subplots(figsize=(8, 3.8), facecolor=bg_color)
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        plt.tight_layout(pad=0)
        return fig
    except Exception:
        return None
