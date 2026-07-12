"""
Tests for src/drilldown.py — specifically verifying the threshold constraints
for segment credibility (baseline_share_pct floor).
"""
import pandas as pd
from src.drilldown import drilldown_dimension


def test_drilldown_dimension_filters_insignificant_trap_segments():
    dates = pd.date_range("2024-01-01", periods=9, freq="W-MON")
    
    records = []
    # 8 baseline weeks: dominant segment has 1000.0 BRL, tiny segment has 20.0 BRL
    for dt in dates[:8]:
        records.append({
            "order_status": "delivered",
            "order_purchase_timestamp": dt,
            "order_revenue_item": 1000.0,
            "segment_name": "dominant"
        })
        records.append({
            "order_status": "delivered",
            "order_purchase_timestamp": dt,
            "order_revenue_item": 20.0,
            "segment_name": "tiny"
        })
        
    # Anomaly week: dominant declines to 500.0, tiny has zero order activity (drops to 0.0)
    records.append({
        "order_status": "delivered",
        "order_purchase_timestamp": dates[-1],
        "order_revenue_item": 500.0,
        "segment_name": "dominant"
    })

    df = pd.DataFrame(records)
    anomaly_week = dates[-1]

    result = drilldown_dimension(df, "segment_name", anomaly_week)

    # Ranks by contribution_pct_of_drop
    tiny_row = result[result["segment"] == "tiny"].iloc[0]
    dominant_row = result[result["segment"] == "dominant"].iloc[0]

    # Verification:
    # 1. Tiny segment has indeed dropped by 100%
    assert tiny_row["segment_pct_change"] == -100.0
    
    # 2. Tiny segment baseline share is < 3% (20 / 1020 * 100 = 1.96%)
    assert tiny_row["baseline_share_pct"] < 3.0
    
    # 3. Tiny segment is NOT flagged as a credible cause despite the -100% drop
    assert tiny_row["is_credible_root_cause"] == False

    # 4. Dominant segment has share > 3% but fell exactly in line with overall average
    # (overall drop is ~50.98%, dominant drop is -50.0%, so decline ratio is ~0.98x < 1.3x)
    assert dominant_row["decline_ratio_vs_overall"] < 1.3
    assert dominant_row["is_credible_root_cause"] == False
