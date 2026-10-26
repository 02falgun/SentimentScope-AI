"""
Tests for PDF report generation and CSV export capabilities.
"""

import pandas as pd
import pytest

from src.sentimentscope.config import DEFAULT_MODEL_PATH, DEFAULT_SAMPLE_DATASET
from src.sentimentscope.data.loading import load_csv
from src.sentimentscope.ml.model_registry import load_model_artifact
from src.sentimentscope.reporting.pdf_report import generate_pdf_report


def test_generate_pdf_report():
    df, _ = load_csv(DEFAULT_SAMPLE_DATASET)
    artifact, _ = load_model_artifact(DEFAULT_MODEL_PATH)

    dist = {
        "counts": {"Positive": 120, "Neutral": 120, "Negative": 120},
        "percentages": {"Positive": 33.3, "Neutral": 33.3, "Negative": 33.3},
        "total": 360,
    }

    pdf_bytes = generate_pdf_report(
        dataset_name="Test Social Media Sample",
        total_records=360,
        sentiment_dist=dist,
        model_artifact=artifact,
        platform_breakdown=pd.DataFrame([
            {"Platform": "Twitter", "Positive": 40, "Neutral": 35, "Negative": 45, "Total": 120},
            {"Platform": "Reddit", "Positive": 38, "Neutral": 42, "Negative": 40, "Total": 120},
        ]),
        top_hashtags=[("#AI", 45), ("#Tech", 30), ("#Fail", 22)],
    )

    assert isinstance(pdf_bytes, (bytes, bytearray))
    assert len(pdf_bytes) > 1000
    # PDF magic header check: %PDF
    assert pdf_bytes.startswith(b"%PDF")
