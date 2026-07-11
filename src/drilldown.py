"""
drilldown.py

Drills into merged_orders.parquet to find which segments of a single
dimension (or combination of two dimensions) are driving a flagged
revenue anomaly week.

NOTE: "payment_type" from the original spec does not exist as a raw
column in merged_orders.parquet — Day 1's build_dataset.py aggregated
payments to payment_type_mode (the most common payment method per
order) BEFORE joining, specifically to avoid fan-out. This script uses
payment_type_mode in its place.

Run from project root with the venv active:
    python -m src.drilldown
"""

from pathlib import Path
import sys
import pandas as pd

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
INPUT_PATH = PROCESSED_DIR / "merged_orders.parquet"

# Same exclusion as build_timeseries.py — see that file for the full
# explanation of why this is the extent of "returns handling" possible
# with this dataset.
EXCLUDED_STATUSES = ["canceled", "unavailable"]

DIMENSIONS = [
    "product_category_name_english",
    "customer_state",
    "seller_state",
    "payment_type_mode",  # substituted for "payment_type" — see module docstring
]

BASELINE_SHARE_THRESHOLD = 3.0     # segment must be at least 3% of baseline revenue
DECLINE_RATIO_THRESHOLD = 1.3      # segment must be dropping >1.3x faster than overall


def _prepare_base(df: pd.DataFrame) -> pd.DataFrame:
    """Exclude canceled/unavailable orders and attach a Monday week_start,
    consistent with build_timeseries.py's bucketing logic."""
    df = df[~df["order_status"].isin(EXCLUDED_STATUSES)].copy()
    df["week_start"] = (
        df["order_purchase_timestamp"]
        - pd.to_timedelta(df["order_purchase_timestamp"].dt.dayofweek, unit="D")
    ).dt.normalize()
    return df


def drilldown_dimension(
    df: pd.DataFrame,
    dimension: str,
    anomaly_week_start,
    baseline_weeks: int = 8,
) -> pd.DataFrame:
    """
    For a single dimension, compares each segment's anomaly-week revenue
    against its trailing baseline average, and ranks segments by their
    DOLLAR contribution to the total revenue drop (not raw % change,
    which can make a tiny segment with a big % swing look falsely
    important).
    """
    anomaly_week_start = pd.Timestamp(anomaly_week_start)
    baseline_start = anomaly_week_start - pd.Timedelta(weeks=baseline_weeks)
    baseline_end = anomaly_week_start - pd.Timedelta(weeks=1)

    base = _prepare_base(df)
    base[dimension] = base[dimension].fillna("(missing)")

    window = base[
        (base["week_start"] >= baseline_start) & (base["week_start"] <= anomaly_week_start)
    ]

    # Anomaly week revenue per segment
    anomaly_rev = (
        window[window["week_start"] == anomaly_week_start]
        .groupby(dimension)["order_revenue_item"]
        .sum()
    )

    # Baseline: per-segment weekly revenue, reindexed to the FULL set of
    # baseline weeks (so a segment with zero orders in some baseline week
    # counts as 0 for that week, not simply absent from the average).
    baseline_pivot = (
        window[(window["week_start"] >= baseline_start) & (window["week_start"] <= baseline_end)]
        .groupby(["week_start", dimension])["order_revenue_item"]
        .sum()
        .unstack(fill_value=0)
    )
    full_baseline_weeks = pd.date_range(baseline_start, baseline_end, freq="W-MON")
    baseline_pivot = baseline_pivot.reindex(full_baseline_weeks, fill_value=0)
    baseline_avg = baseline_pivot.mean(axis=0)

    all_segments = sorted(set(anomaly_rev.index) | set(baseline_avg.index))
    result = pd.DataFrame({"segment": all_segments})
    result["dimension"] = dimension
    result["anomaly_week_revenue"] = result["segment"].map(anomaly_rev).fillna(0.0)
    result["baseline_avg_revenue"] = result["segment"].map(baseline_avg).fillna(0.0)

    total_baseline = result["baseline_avg_revenue"].sum()
    result["baseline_share_pct"] = (
        result["baseline_avg_revenue"] / total_baseline * 100 if total_baseline else 0.0
    )

    result["dollar_change"] = result["anomaly_week_revenue"] - result["baseline_avg_revenue"]
    total_drop = result["dollar_change"].sum()  # negative if revenue fell overall

    # contribution_pct_of_drop: this segment's dollar change as a % of the
    # TOTAL dollar change. Ranking by this (not % change) is what lets a
    # big, moderately-declining segment outrank a tiny segment that
    # cratered 90% but barely mattered in dollar terms.
    result["contribution_pct_of_drop"] = (
        result["dollar_change"] / total_drop * 100 if total_drop != 0 else float("nan")
    )

    result["segment_pct_change"] = result.apply(
        lambda r: (r["dollar_change"] / r["baseline_avg_revenue"] * 100)
        if r["baseline_avg_revenue"] > 1e-6
        else float("nan"),
        axis=1,
    )

    overall_pct_change = total_drop / total_baseline * 100 if total_baseline else float("nan")
    result["decline_ratio_vs_overall"] = (
        result["segment_pct_change"] / overall_pct_change if overall_pct_change else float("nan")
    )

    result["is_credible_root_cause"] = (
        (result["baseline_share_pct"] >= BASELINE_SHARE_THRESHOLD)
        & (result["decline_ratio_vs_overall"] > DECLINE_RATIO_THRESHOLD)
        & (result["dollar_change"] < 0)
    )

    return result.sort_values("contribution_pct_of_drop", ascending=False).reset_index(drop=True)


def run_full_drilldown(df: pd.DataFrame, anomaly_week_start, baseline_weeks: int = 8) -> dict:
    """Runs drilldown_dimension across all 4 dimensions, returns a dict keyed by dimension name."""
    return {
        dim: drilldown_dimension(df, dim, anomaly_week_start, baseline_weeks)
        for dim in DIMENSIONS
    }


def find_combination_effects(
    df: pd.DataFrame,
    anomaly_week_start,
    dim1: str,
    dim2: str,
    baseline_weeks: int = 8,
) -> pd.DataFrame:
    """
    Same analysis as drilldown_dimension, but grouped by (dim1, dim2)
    jointly instead of one dimension at a time.

    WHY THIS CATCHES THINGS SINGLE-DIMENSION ANALYSIS MISSES:
    A drop can be concentrated in one specific combination — e.g. a
    particular category paid via a particular payment method — while
    looking unremarkable in either dimension checked alone. Checking
    "category" alone averages that combination's drop together with the
    SAME category's other, unaffected payment methods, diluting the
    signal. Checking "payment_type" alone dilutes it the opposite way,
    averaging it with every other category using that payment method.
    Only grouping jointly isolates the specific cell where the drop is
    actually concentrated — the single-dimension views can each look
    calm even when a real, sharp, localized problem exists.
    """
    anomaly_week_start = pd.Timestamp(anomaly_week_start)
    baseline_start = anomaly_week_start - pd.Timedelta(weeks=baseline_weeks)
    baseline_end = anomaly_week_start - pd.Timedelta(weeks=1)

    base = _prepare_base(df)
    base[dim1] = base[dim1].fillna("(missing)")
    base[dim2] = base[dim2].fillna("(missing)")
    base["_combo"] = list(zip(base[dim1], base[dim2]))

    window = base[
        (base["week_start"] >= baseline_start) & (base["week_start"] <= anomaly_week_start)
    ]

    anomaly_rev = (
        window[window["week_start"] == anomaly_week_start]
        .groupby("_combo")["order_revenue_item"]
        .sum()
    )

    baseline_pivot = (
        window[(window["week_start"] >= baseline_start) & (window["week_start"] <= baseline_end)]
        .groupby(["week_start", "_combo"])["order_revenue_item"]
        .sum()
        .unstack(fill_value=0)
    )
    full_baseline_weeks = pd.date_range(baseline_start, baseline_end, freq="W-MON")
    baseline_pivot = baseline_pivot.reindex(full_baseline_weeks, fill_value=0)
    baseline_avg = baseline_pivot.mean(axis=0)

    all_combos = sorted(set(anomaly_rev.index) | set(baseline_avg.index))
    result = pd.DataFrame({"segment": all_combos})
    result[dim1] = result["segment"].apply(lambda c: c[0])
    result[dim2] = result["segment"].apply(lambda c: c[1])
    result["dimension"] = f"{dim1} x {dim2}"
    result["anomaly_week_revenue"] = result["segment"].map(anomaly_rev).fillna(0.0)
    result["baseline_avg_revenue"] = result["segment"].map(baseline_avg).fillna(0.0)

    total_baseline = result["baseline_avg_revenue"].sum()
    result["baseline_share_pct"] = (
        result["baseline_avg_revenue"] / total_baseline * 100 if total_baseline else 0.0
    )

    result["dollar_change"] = result["anomaly_week_revenue"] - result["baseline_avg_revenue"]
    total_drop = result["dollar_change"].sum()

    result["contribution_pct_of_drop"] = (
        result["dollar_change"] / total_drop * 100 if total_drop != 0 else float("nan")
    )

    result["segment_pct_change"] = result.apply(
        lambda r: (r["dollar_change"] / r["baseline_avg_revenue"] * 100)
        if r["baseline_avg_revenue"] > 1e-6
        else float("nan"),
        axis=1,
    )

    overall_pct_change = total_drop / total_baseline * 100 if total_baseline else float("nan")
    result["decline_ratio_vs_overall"] = (
        result["segment_pct_change"] / overall_pct_change if overall_pct_change else float("nan")
    )

    result["is_credible_root_cause"] = (
        (result["baseline_share_pct"] >= BASELINE_SHARE_THRESHOLD)
        & (result["decline_ratio_vs_overall"] > DECLINE_RATIO_THRESHOLD)
        & (result["dollar_change"] < 0)
    )

    result = result.drop(columns=["segment"])
    return result.sort_values("contribution_pct_of_drop", ascending=False).reset_index(drop=True)


def main():
    # NOTE on findings from real anomaly weeks tested against this module:
    #
    # Week of 2018-08-27 (dataset's final week): no concentrated root cause
    # found — every dimension declines uniformly (~90-98%), consistent with
    # a platform-wide data collection cutoff rather than a business-specific
    # problem.
    #
    # Week of 2018-05-21: THREE credible single-dimension root causes found
    # (customer_state=RJ, customer_state=PR, category=sports_leisure), plus
    # TWO credible combination effects (category=auto x credit_card,
    # category=baby x credit_card). This timing is consistent with the
    # documented May 2018 Brazilian truckers' strike, which disrupted
    # logistics nationwide — however, this pipeline only identifies WHICH
    # segments declined disproportionately; the causal link to the strike
    # is an external hypothesis based on real-world context, not something
    # derived from the data itself.

    df = pd.read_parquet(INPUT_PATH)

    if len(sys.argv) > 1:
        anomaly_week_start = pd.Timestamp(sys.argv[1])
    else:
        anomaly_week_start = pd.Timestamp("2018-08-27")

    print(f"--- Full drilldown for week of {anomaly_week_start.date()} ---\n")
    results = run_full_drilldown(df, anomaly_week_start)

    all_credible = []
    for dim, res in results.items():
        credible = res[res["is_credible_root_cause"]]
        all_credible.append(credible)
        print(f"[{dim}] top 3 segments by contribution_pct_of_drop:")
        print(res.head(3)[
            ["segment", "baseline_share_pct", "contribution_pct_of_drop",
             "segment_pct_change", "decline_ratio_vs_overall", "is_credible_root_cause"]
        ].to_string(index=False))
        print()

    all_credible_df = pd.concat(all_credible, ignore_index=True) if all_credible else pd.DataFrame()

    print("--- Top 3 CREDIBLE root causes across all single dimensions ---")
    if all_credible_df.empty:
        print("No segment met the credibility bar (share >= 3%, decline ratio > 1.3x, actually dropped).")
    else:
        top3 = all_credible_df.sort_values("contribution_pct_of_drop", ascending=False).head(3)
        print(top3[
            ["dimension", "segment", "baseline_share_pct", "contribution_pct_of_drop",
             "segment_pct_change", "decline_ratio_vs_overall"]
        ].to_string(index=False))

    print("\n--- Combination effect: product_category_name_english x payment_type_mode ---")
    combo = find_combination_effects(
        df, anomaly_week_start, "product_category_name_english", "payment_type_mode"
    )
    credible_combo = combo[combo["is_credible_root_cause"]]
    if credible_combo.empty:
        print("No credible combination effect found. Top 3 by contribution shown for reference:")
        print(combo.head(3)[
            ["product_category_name_english", "payment_type_mode", "baseline_share_pct",
             "contribution_pct_of_drop", "segment_pct_change", "decline_ratio_vs_overall"]
        ].to_string(index=False))
    else:
        print(credible_combo.head(3)[
            ["product_category_name_english", "payment_type_mode", "baseline_share_pct",
             "contribution_pct_of_drop", "segment_pct_change", "decline_ratio_vs_overall"]
        ].to_string(index=False))


if __name__ == "__main__":
    main()
