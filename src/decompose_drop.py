"""
decompose_drop.py

Decomposes a revenue drop in a given anomaly week into a volume effect
and an AOV (average order value) effect, using an exact additive
decomposition:

    Revenue = Volume x AOV

    volume_effect = (volume_now - volume_baseline) * aov_baseline
    aov_effect     = (aov_now - aov_baseline) * volume_now

    volume_effect + aov_effect == revenue_now - revenue_baseline  (exact, no residual)

Run from project root with the venv active:
    python -m src.decompose_drop
"""

from pathlib import Path
import pandas as pd

from src.anomaly_detector import detect_revenue_anomalies

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
INPUT_PATH = PROCESSED_DIR / "weekly_revenue.parquet"

ROLLING_WINDOW = 8
MIN_PERIODS = 4

# Thresholds for classifying which effect "drives" the drop.
# An effect is considered "significant" if its magnitude is at least
# this fraction of the total absolute revenue change.
SIGNIFICANCE_THRESHOLD = 0.20


def _trailing_baseline(series: pd.Series) -> pd.Series:
    """Shift-by-1 trailing rolling mean, matching anomaly_detector's logic
    so the baseline here is consistent with what flagged the anomaly."""
    return series.shift(1).rolling(window=ROLLING_WINDOW, min_periods=MIN_PERIODS).mean()


def decompose_revenue_drop(weekly: pd.DataFrame, week_start: pd.Timestamp) -> dict:
    """
    Decomposes the revenue change at `week_start` into volume and AOV
    effects, relative to the trailing 8-week baseline (excluding the
    week itself, consistent with anomaly_detector.py).

    Returns a dict with raw values, % changes, effect sizes, effect
    %-contributions, and a classification label.
    """
    df = weekly.sort_values("week_start").reset_index(drop=True)

    df["volume_baseline"] = _trailing_baseline(df["order_count"])
    df["aov_baseline"] = _trailing_baseline(df["avg_order_value"])
    # revenue_baseline MUST be derived as volume_baseline * aov_baseline,
    # NOT as an independent rolling mean of the revenue column. Revenue =
    # Volume x AOV holds exactly for every individual week, but the mean
    # of several weeks' revenue is NOT generally equal to (mean volume x
    # mean AOV) — averaging a product isn't the same as multiplying two
    # separate averages, unless volume and AOV are uncorrelated across
    # those weeks. Deriving revenue_baseline this way instead of
    # independently guarantees volume_effect + aov_effect sums EXACTLY
    # to the true revenue change, with zero residual.
    df["revenue_baseline"] = df["volume_baseline"] * df["aov_baseline"]

    row = df[df["week_start"] == week_start]
    if row.empty:
        raise ValueError(f"week_start {week_start} not found in weekly data.")
    row = row.iloc[0]

    volume_now = row["order_count"]
    volume_baseline = row["volume_baseline"]
    aov_now = row["avg_order_value"]
    aov_baseline = row["aov_baseline"]
    revenue_now = row["revenue"]
    revenue_baseline = row["revenue_baseline"]

    if pd.isna(volume_baseline) or pd.isna(aov_baseline) or pd.isna(revenue_baseline):
        return {
            "week_start": week_start,
            "has_baseline": False,
            "note": "Insufficient trailing history to decompose this week.",
        }

    # --- Step 3: exact additive decomposition ---
    volume_effect = (volume_now - volume_baseline) * aov_baseline
    aov_effect = (aov_now - aov_baseline) * volume_now
    total_effect = volume_effect + aov_effect  # should equal revenue_now - revenue_baseline

    revenue_change = revenue_now - revenue_baseline
    revenue_pct_change = (revenue_change / revenue_baseline * 100) if revenue_baseline else float("nan")
    volume_pct_change = (
        (volume_now - volume_baseline) / volume_baseline * 100 if volume_baseline else float("nan")
    )
    aov_pct_change = (aov_now - aov_baseline) / aov_baseline * 100 if aov_baseline else float("nan")

    # % contribution of each effect to the total change (guard divide-by-zero
    # for the rare case total_effect is ~0 despite both sub-effects nonzero,
    # i.e. a perfectly offsetting week).
    if abs(total_effect) > 1e-6:
        volume_contrib_pct = volume_effect / total_effect * 100
        aov_contrib_pct = aov_effect / total_effect * 100
    else:
        volume_contrib_pct = float("nan")
        aov_contrib_pct = float("nan")

    # --- Step 4: classification ---
    volume_significant = abs(volume_effect) >= SIGNIFICANCE_THRESHOLD * abs(total_effect) if total_effect else False
    aov_significant = abs(aov_effect) >= SIGNIFICANCE_THRESHOLD * abs(total_effect) if total_effect else False
    same_direction = (volume_effect >= 0) == (aov_effect >= 0)

    if volume_significant and aov_significant:
        classification = "compounding" if same_direction else "offsetting"
    elif volume_significant:
        classification = "volume-driven"
    elif aov_significant:
        classification = "AOV-driven"
    else:
        classification = "negligible-change"

    return {
        "week_start": week_start,
        "has_baseline": True,
        "revenue_now": revenue_now,
        "revenue_baseline": revenue_baseline,
        "revenue_pct_change": revenue_pct_change,
        "volume_now": volume_now,
        "volume_baseline": volume_baseline,
        "volume_pct_change": volume_pct_change,
        "aov_now": aov_now,
        "aov_baseline": aov_baseline,
        "aov_pct_change": aov_pct_change,
        "volume_effect": volume_effect,
        "aov_effect": aov_effect,
        "total_effect": total_effect,
        "volume_contrib_pct": volume_contrib_pct,
        "aov_contrib_pct": aov_contrib_pct,
        "classification": classification,
    }


def main():
    weekly = pd.read_parquet(INPUT_PATH)
    anomalies = detect_revenue_anomalies(weekly)
    flagged = anomalies[anomalies["severity"].isin(["moderate_anomaly", "severe_anomaly"])]

    if flagged.empty:
        print("No anomaly weeks to decompose.")
        return

    print(f"--- Decomposing {len(flagged)} anomaly week(s) ---\n")
    for _, arow in flagged.iterrows():
        result = decompose_revenue_drop(weekly, arow["week_start"])

        if not result["has_baseline"]:
            print(f"Week of {arow['week_start'].date()}: insufficient history to decompose.")
            continue

        print(
            f"Week of {result['week_start'].date()}: "
            f"revenue {result['revenue_pct_change']:+.0f}%, "
            f"classified as {result['classification']} "
            f"(AOV {result['aov_pct_change']:+.0f}%, volume {result['volume_pct_change']:+.0f}%)"
        )


if __name__ == "__main__":
    main()
