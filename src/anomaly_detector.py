"""
anomaly_detector.py

Detects revenue anomalies in the weekly revenue series using a rolling
z-score approach: each week's revenue is compared against a trailing
baseline (mean + std) of the prior 8 weeks, EXCLUDING the week itself.

Run from project root with the venv active:
    python -m src.anomaly_detector
"""

from pathlib import Path
import pandas as pd
import numpy as np

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
INPUT_PATH = PROCESSED_DIR / "weekly_revenue.parquet"

ROLLING_WINDOW = 8
MIN_PERIODS = 4
MODERATE_THRESHOLD = -1.5
SEVERE_THRESHOLD = -2.5

# Any std below this is treated as "effectively zero" to avoid a
# divide-by-zero / near-infinite z-score on a flat baseline period.
STD_EPSILON = 1e-6


def detect_revenue_anomalies(weekly: pd.DataFrame) -> pd.DataFrame:
    """
    Flags weeks whose revenue is anomalously LOW relative to their own
    trailing baseline.

    Parameters
    ----------
    weekly : DataFrame with columns week_start, revenue (at minimum),
             sorted chronologically ascending.

    Returns
    -------
    DataFrame with columns:
        week_start, revenue, rolling_mean, rolling_std, z_score,
        severity, has_baseline
    """
    df = weekly[["week_start", "revenue"]].copy()
    df = df.sort_values("week_start").reset_index(drop=True)

    # --- Step 2: exclude the current week from its own baseline ---
    # We shift the revenue series forward by 1 BEFORE computing the
    # rolling mean/std, so that when we compute stats "as of" week N,
    # they only ever see weeks N-8 .. N-1 — never week N itself.
    #
    # WHY THIS MATTERS: if week N's own revenue value were included in
    # its own baseline window, a genuine anomaly (e.g. a severe crash)
    # would directly pull the mean down and INFLATE the std toward that
    # same crashed value. The baseline would partially "absorb" the very
    # anomaly you're trying to detect, shrinking the z-score and making
    # real anomalies look milder than they are — in the worst case,
    # masking them entirely. Shifting by 1 guarantees the baseline is
    # computed purely from *prior* weeks, so it reflects what was
    # "normal" going into that week, uncontaminated by the week's own
    # outcome.
    shifted_revenue = df["revenue"].shift(1)

    df["rolling_mean"] = shifted_revenue.rolling(
        window=ROLLING_WINDOW, min_periods=MIN_PERIODS
    ).mean()
    df["rolling_std"] = shifted_revenue.rolling(
        window=ROLLING_WINDOW, min_periods=MIN_PERIODS
    ).std()

    # --- Step 5: mark weeks with insufficient history ---
    # A week only "has a baseline" if the rolling window found at least
    # MIN_PERIODS prior weeks to compute mean/std from. Early weeks (the
    # first 4, given shift(1) + min_periods=4) will have NaN rolling
    # stats — rather than let those propagate into misleading NaN or
    # inf z-scores, we explicitly flag them as having no baseline and
    # exclude them from anomaly flagging entirely.
    df["has_baseline"] = df["rolling_mean"].notna() & df["rolling_std"].notna()

    # --- Step 6: guard against divide-by-zero on a near-flat baseline ---
    # If the trailing 8 weeks were essentially flat (e.g. the 2016
    # launch period, mostly zeros), rolling_std can be 0 or extremely
    # close to 0. Dividing by that would produce inf or a wildly
    # oversized z-score for even a tiny revenue change. We treat any
    # std below STD_EPSILON as "no meaningful variation to compare
    # against" and skip z-score computation for those weeks instead of
    # crashing or producing a nonsensical spike.
    safe_std = df["rolling_std"].where(df["rolling_std"] > STD_EPSILON)

    # --- Step 3: z-score ---
    df["z_score"] = (df["revenue"] - df["rolling_mean"]) / safe_std

    # Weeks with no baseline, or where std was too close to zero to
    # trust, get an explicit NaN z-score rather than a misleading number.
    df.loc[~df["has_baseline"], "z_score"] = np.nan

    # --- Step 4: severity flags ---
    def classify(row):
        if not row["has_baseline"] or pd.isna(row["z_score"]):
            return "no_baseline"
        if row["z_score"] < SEVERE_THRESHOLD:
            return "severe_anomaly"
        if row["z_score"] < MODERATE_THRESHOLD:
            return "moderate_anomaly"
        return "normal"

    df["severity"] = df.apply(classify, axis=1)

    return df[
        ["week_start", "revenue", "rolling_mean", "rolling_std", "z_score", "severity", "has_baseline"]
    ]


def main():
    weekly = pd.read_parquet(INPUT_PATH)
    result = detect_revenue_anomalies(weekly)

    flagged = result[result["severity"].isin(["moderate_anomaly", "severe_anomaly"])].copy()

    if flagged.empty:
        print("No anomalies flagged.")
        return

    flagged["pct_drop_from_baseline"] = (
        (flagged["revenue"] - flagged["rolling_mean"]) / flagged["rolling_mean"] * 100
    )

    print(f"--- Flagged Anomalies ({len(flagged)} weeks) ---\n")
    for _, row in flagged.iterrows():
        print(
            f"{row['week_start'].date()}  |  severity={row['severity']:<16}  "
            f"z={row['z_score']:.2f}  |  revenue={row['revenue']:,.2f}  "
            f"baseline_mean={row['rolling_mean']:,.2f}  |  "
            f"pct_drop={row['pct_drop_from_baseline']:.1f}%"
        )

    print(f"\n--- Full result shape ---")
    print(result.shape)


if __name__ == "__main__":
    main()
