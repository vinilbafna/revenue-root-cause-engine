"""
Tests for src/decompose_drop.py — specifically verifying the exact two-term
additive decomposition of revenue changes.
"""
import pytest
import pandas as pd
from src.decompose_drop import decompose_revenue_drop


def test_decompose_revenue_drop_is_exact_and_compounding():
    # 8 stable baseline weeks followed by 1 anomaly week
    dates = pd.date_range("2024-01-01", periods=9, freq="W-MON")
    
    order_counts = [100] * 8 + [50]
    avg_order_values = [10.0] * 8 + [5.0]
    # Revenue = Volume x AOV
    revenues = [v * a for v, a in zip(order_counts, avg_order_values)]
    
    weekly = pd.DataFrame({
        "week_start": dates,
        "order_count": order_counts,
        "avg_order_value": avg_order_values,
        "revenue": revenues,
    })

    anomaly_week = dates[-1]
    result = decompose_revenue_drop(weekly, anomaly_week)

    assert result["has_baseline"] is True
    
    revenue_now = result["revenue_now"]
    revenue_baseline = result["revenue_baseline"]
    # Ground-truth check: baseline weeks are all order_count=100, avg_order_value=10.0,
    # so true baseline revenue must be exactly 100 * 10.0 = 1000.0. Catches bugs where
    # revenue_baseline drifts from genuinely being volume_baseline * aov_baseline.
    assert revenue_baseline == pytest.approx(1000.0)
    volume_effect = result["volume_effect"]
    aov_effect = result["aov_effect"]
    total_effect = result["total_effect"]
    
    # 1. Exact additive decomposition check
    expected_change = revenue_now - revenue_baseline
    assert expected_change == pytest.approx(total_effect)
    assert (volume_effect + aov_effect) == pytest.approx(expected_change)
    
    # 2. Classification check: Compounding because both decreased significantly
    assert result["classification"] == "compounding"
    assert result["volume_pct_change"] == -50.0
    assert result["aov_pct_change"] == -50.0
