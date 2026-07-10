"""
build_timeseries.py

Builds a weekly revenue time series from data/processed/merged_orders.parquet,
for use in the volume-vs-AOV decomposition and downstream root-cause analysis.

Run from project root with the venv active:
    python -m src.build_timeseries
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
INPUT_PATH = PROCESSED_DIR / "merged_orders.parquet"
OUTPUT_PATH = PROCESSED_DIR / "weekly_revenue.parquet"
PLOT_PATH = PROCESSED_DIR / "weekly_revenue_plot.png"

# Order statuses excluded from revenue.
#
# IMPORTANT LIMITATION: Olist provides no post-delivery returns/refunds
# dataset at all. The only signal we have for "this order didn't result
# in real revenue" is order_status. Excluding 'canceled' and 'unavailable'
# is therefore the FULL EXTENT of returns/cancellation handling possible
# with this data — it does not capture orders that were delivered and
# later returned or refunded, because that information simply does not
# exist in any of the 7 source tables. This is a real analytical gap in
# the dataset, not an oversight in this script, and should be stated
# explicitly whenever revenue totals from this pipeline are presented.
EXCLUDED_STATUSES = ["canceled", "unavailable"]


def load_and_filter() -> pd.DataFrame:
    df = pd.read_parquet(INPUT_PATH)

    before = len(df)
    df = df[~df["order_status"].isin(EXCLUDED_STATUSES)].copy()
    after = len(df)
    print(f"Excluded {before - after:,} rows with order_status in {EXCLUDED_STATUSES} "
          f"({before:,} -> {after:,} rows)")

    return df


def inspect_date_range(df: pd.DataFrame) -> None:
    """
    Step 3: report the min/max purchase timestamp and month-by-month order
    counts, so partial first/last months are visible before any trimming.

    We use order_purchase_timestamp — NOT any delivery date — as the single
    date field for all revenue analysis. Purchase timestamp is when revenue
    is actually booked (the transaction happened). Delivery dates reflect
    logistics/fulfillment performance, which varies for reasons unrelated
    to revenue generation (carrier delays, distance, warehouse backlogs).
    Using a delivery date to bucket revenue would contaminate the revenue
    signal with fulfillment noise, making it impossible to cleanly separate
    "did we sell less" from "did shipping take longer that month."
    """
    min_date = df["order_purchase_timestamp"].min()
    max_date = df["order_purchase_timestamp"].max()
    print(f"\nPurchase timestamp range: {min_date} to {max_date}")

    monthly_counts = (
        df.set_index("order_purchase_timestamp")
        .resample("MS")["order_id"]
        .nunique()
    )
    print("\nMonth-by-month distinct order counts:")
    print(monthly_counts.to_string())


def build_weekly_series(df: pd.DataFrame) -> pd.DataFrame:
    """
    Steps 4-6: resample to weekly (week starting Monday), trim partial
    edge weeks, and compute revenue, order volume, and AOV per week.
    """
    df = df.copy()
    # Monday of the week each purchase falls in.
    df["week_start"] = (
        df["order_purchase_timestamp"]
        - pd.to_timedelta(df["order_purchase_timestamp"].dt.dayofweek, unit="D")
    ).dt.normalize()

    weekly = (
        df.groupby("week_start")
        .agg(
            revenue=("order_revenue_item", "sum"),
            order_count=("order_id", "nunique"),
        )
        .reset_index()
        .sort_values("week_start")
        .reset_index(drop=True)
    )
    weekly["avg_order_value"] = weekly["revenue"] / weekly["order_count"]

    # --- Step 5: detect and trim partial first/last weeks ---
    actual_min = df["order_purchase_timestamp"].min()
    actual_max = df["order_purchase_timestamp"].max()

    first_week_start = weekly["week_start"].iloc[0]
    last_week_start = weekly["week_start"].iloc[-1]
    last_week_end = last_week_start + pd.Timedelta(days=6)

    trimmed_rows = []

    # First week is partial if the data's true minimum timestamp is later
    # than that week's Monday (i.e. we don't have all 7 days of that week).
    if actual_min > first_week_start:
        trimmed_rows.append(
            f"Trimmed first week starting {first_week_start.date()}: "
            f"data only starts {actual_min.date()} ({actual_min.strftime('%A')}), "
            f"so this week has fewer than 7 days of purchase activity."
        )
        weekly = weekly[weekly["week_start"] != first_week_start]

    # Last week is partial if the data's true maximum timestamp is earlier
    # than that week's Sunday (i.e. the week got cut off before completing).
    if actual_max < last_week_end:
        trimmed_rows.append(
            f"Trimmed last week starting {last_week_start.date()}: "
            f"data ends {actual_max.date()} ({actual_max.strftime('%A')}), "
            f"before that week's Sunday ({last_week_end.date()}), "
            f"so this week has fewer than 7 days of purchase activity."
        )
        weekly = weekly[weekly["week_start"] != last_week_start]

    print("\n--- Weekly Trimming ---")
    if trimmed_rows:
        for line in trimmed_rows:
            print(line)
    else:
        print("No partial edge weeks detected — no trimming needed.")

    weekly = weekly.reset_index(drop=True)
    return weekly


def fill_missing_weeks(weekly: pd.DataFrame) -> pd.DataFrame:
    """
    Reindex to a complete Monday-starting weekly calendar between the min
    and max week_start, filling any missing weeks with explicit zeros.

    Olist's early launch period (late 2016) had weeks with zero order
    activity entirely. Without this step, those weeks would be silently
    ABSENT from the series (not present with revenue=0), which would
    corrupt any downstream week-over-week % change, rolling average, or
    plot — those would silently treat non-adjacent weeks as adjacent.
    """
    full_calendar = pd.date_range(
        weekly["week_start"].min(), weekly["week_start"].max(), freq="W-MON"
    )
    weekly = weekly.set_index("week_start").reindex(full_calendar)
    weekly.index.name = "week_start"

    missing_mask = weekly["revenue"].isna()
    n_missing = missing_mask.sum()
    if n_missing > 0:
        print(f"\nFilled {n_missing} missing week(s) with explicit zeros "
              f"(no order activity that week):")
        print(weekly[missing_mask].index.tolist())

    weekly["revenue"] = weekly["revenue"].fillna(0.0)
    weekly["order_count"] = weekly["order_count"].fillna(0).astype(int)
    # avg_order_value is undefined (0/0) for a zero-order week — leave as 0
    # rather than NaN, so downstream code doesn't need extra NaN-handling,
    # but be aware this 0 means "no orders," not "orders averaged to zero."
    weekly["avg_order_value"] = weekly["avg_order_value"].fillna(0.0)

    return weekly.reset_index()


def plot_weekly_revenue(weekly: pd.DataFrame) -> None:
    plt.figure(figsize=(12, 5))
    plt.plot(weekly["week_start"], weekly["revenue"], marker="o", markersize=3)
    plt.title("Weekly Revenue (order_revenue_item, purchase-date basis)")
    plt.xlabel("Week starting (Monday)")
    plt.ylabel("Revenue (BRL)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=150)
    print(f"\nSaved plot to: {PLOT_PATH}")
    plt.show()


def main():
    df = load_and_filter()
    inspect_date_range(df)
    weekly = build_weekly_series(df)
    weekly = fill_missing_weeks(weekly)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    weekly.to_parquet(OUTPUT_PATH, index=False)
    print(f"\nSaved weekly revenue series to: {OUTPUT_PATH}")

    print("\n--- Final weekly series shape ---")
    print(weekly.shape)
    print("\n--- Preview ---")
    print(weekly.head())
    print(weekly.tail())

    plot_weekly_revenue(weekly)


if __name__ == "__main__":
    main()
