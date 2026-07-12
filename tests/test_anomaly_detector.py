"""
Tests for src/anomaly_detector.py — verifying the robustness of anomaly
flagging, including flat baseline protection and minimum history checks.
"""
import pandas as pd
import numpy as np
from src.anomaly_detector import detect_revenue_anomalies


def test_detect_revenue_anomalies_flat_baseline_does_not_crash():
    dates = pd.date_range("2024-01-01", periods=10, freq="W-MON")
    # All 10 weeks have exactly the same flat revenue
    weekly = pd.DataFrame({
        "week_start": dates,
        "revenue": [100.0] * 10
    })

    result = detect_revenue_anomalies(weekly)

    # 1. Check that the script doesn't crash and yiellds valid rows
    assert len(result) == 10
    
    # 2. Check that no z-score is infinite or -infinite
    assert not np.isinf(result["z_score"]).any()
    
    # 3. For weeks with a baseline, because standard deviation was zero,
    # the z_score should be NaN and severity should be "no_baseline"
    weeks_with_baseline = result[result["has_baseline"]]
    for _, row in weeks_with_baseline.iterrows():
        assert pd.isna(row["z_score"])
        assert row["severity"] == "no_baseline"


def test_detect_revenue_anomalies_short_history_is_not_flagged():
    # Only 3 weeks of history (below MIN_PERIODS = 4)
    dates = pd.date_range("2024-01-01", periods=3, freq="W-MON")
    weekly = pd.DataFrame({
        "week_start": dates,
        "revenue": [100.0, 10.0, 100.0]  # Even with a massive middle drop, it shouldn't flag
    })

    result = detect_revenue_anomalies(weekly)

    # 1. No rows should have a baseline since history length < 4 weeks (MIN_PERIODS)
    assert not result["has_baseline"].any()

    # 2. All rows should be classified as "no_baseline"
    assert (result["severity"] == "no_baseline").all()
    assert result["z_score"].isna().all()
