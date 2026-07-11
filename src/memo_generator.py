"""
memo_generator.py

Combines anomaly_detector.py, decompose_drop.py, and drilldown.py into a
plain-English analyst memo for every flagged anomaly week.

Run from project root with the venv active:
    python -m src.memo_generator
"""

import os
from pathlib import Path
import pandas as pd

from typing import Optional

from src.anomaly_detector import detect_revenue_anomalies
from src.decompose_drop import decompose_revenue_drop
from src.drilldown import run_full_drilldown, find_combination_effects, DIMENSIONS

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
WEEKLY_PATH = PROCESSED_DIR / "weekly_revenue.parquet"
MERGED_PATH = PROCESSED_DIR / "merged_orders.parquet"


def _find_best_credible_cause(drilldown_results: dict, combo_result: pd.DataFrame) -> Optional[dict]:
    """
    Scans all single-dimension drilldown results AND the combination
    result, returns the single strongest credible root cause (by
    contribution_pct_of_drop), or None if nothing is credible anywhere.
    """
    candidates = []

    for dim, df in drilldown_results.items():
        credible = df[df["is_credible_root_cause"]]
        for _, row in credible.iterrows():
            candidates.append({
                "kind": "single",
                "dimension": dim,
                "segment": row["segment"],
                "baseline_share_pct": row["baseline_share_pct"],
                "contribution_pct_of_drop": row["contribution_pct_of_drop"],
                "segment_pct_change": row["segment_pct_change"],
                "decline_ratio_vs_overall": row["decline_ratio_vs_overall"],
            })

    if combo_result is not None and not combo_result.empty:
        credible_combo = combo_result[combo_result["is_credible_root_cause"]]
        for _, row in credible_combo.iterrows():
            dim_cols = [c for c in combo_result.columns if c not in (
                "dimension", "anomaly_week_revenue", "baseline_avg_revenue",
                "baseline_share_pct", "dollar_change", "contribution_pct_of_drop",
                "segment_pct_change", "decline_ratio_vs_overall", "is_credible_root_cause"
            )]
            combo_label = " x ".join(f"{c}={row[c]}" for c in dim_cols)
            candidates.append({
                "kind": "combination",
                "dimension": row["dimension"],
                "segment": combo_label,
                "baseline_share_pct": row["baseline_share_pct"],
                "contribution_pct_of_drop": row["contribution_pct_of_drop"],
                "segment_pct_change": row["segment_pct_change"],
                "decline_ratio_vs_overall": row["decline_ratio_vs_overall"],
            })

    if not candidates:
        return None

    return max(candidates, key=lambda c: c["contribution_pct_of_drop"])


def generate_memo(anomaly_row: pd.Series, decomposition_result: dict, drilldown_results: dict,
                   combo_result: pd.DataFrame = None) -> str:
    """
    Produces a plain-English analyst memo string for a single anomaly week.

    Structure:
        1. Header: week, severity, % revenue drop
        2. Root cause paragraph: volume-vs-AOV classification + top
           credible segment cause (or explicit "no concentrated cause")
        3. Recommendation line, tailored to the finding
    """
    week = anomaly_row["week_start"]
    severity_label = "SEVERE" if anomaly_row["severity"] == "severe_anomaly" else "MODERATE"
    pct_change = decomposition_result["revenue_pct_change"]

    header = (
        f"REVENUE ANOMALY MEMO — Week of {week.date()}\n"
        f"Severity: {severity_label}  |  Revenue change vs. 8-week baseline: {pct_change:+.1f}%"
    )

    classification = decomposition_result["classification"]
    vol_pct = decomposition_result["volume_pct_change"]
    aov_pct = decomposition_result["aov_pct_change"]

    def _describe_secondary(pct: float) -> str:
        """Describes a secondary effect's magnitude honestly, instead of
        always calling it 'stable' just because it wasn't the dominant
        contributor — a week can be 'volume-driven' while AOV still
        moved substantially on its own."""
        if abs(pct) < 5:
            return f"held comparatively stable ({pct:+.1f}%)"
        else:
            return f"also declined meaningfully ({pct:+.1f}%), though less than the primary driver"

    classification_sentence = {
        "volume-driven": f"This decline was driven primarily by fewer orders (order volume {vol_pct:+.1f}%); average order value {_describe_secondary(aov_pct)}.",
        "AOV-driven": f"This decline was driven primarily by a drop in average order value ({aov_pct:+.1f}%); order volume {_describe_secondary(vol_pct)}.",
        "compounding": f"This decline was compounded by both fewer orders ({vol_pct:+.1f}%) and lower average order value ({aov_pct:+.1f}%) moving in the same direction together.",
        "offsetting": f"Order volume ({vol_pct:+.1f}%) and average order value ({aov_pct:+.1f}%) moved in opposite directions, partially offsetting each other, yet revenue still declined overall.",
        "negligible-change": f"Neither order volume ({vol_pct:+.1f}%) nor average order value ({aov_pct:+.1f}%) moved by a meaningful amount on their own.",
    }.get(classification, f"Volume changed {vol_pct:+.1f}% and average order value changed {aov_pct:+.1f}%.")

    best_cause = _find_best_credible_cause(drilldown_results, combo_result)

    if best_cause is None:
        cause_sentence = (
            "No single segment (by category, customer state, seller state, or payment method, "
            "individually or in combination) accounts for a disproportionate share of this decline. "
            "Every segment examined fell by roughly the same proportion as the overall week. "
            "This points to a BROAD-BASED DECLINE WITH NO CONCENTRATED CAUSE — a real, valid finding "
            "in its own right, not an inconclusive result."
        )
        recommendation = (
            "Recommendation: investigate platform-wide or external explanations (site availability, "
            "a payment gateway issue, a data collection boundary, or a market-wide event) rather than "
            "auditing any single category, region, or seller — the data does not support attributing "
            "this to a specific segment."
        )
    else:
        kind_phrase = (
            f"the combination of {best_cause['segment']}"
            if best_cause["kind"] == "combination"
            else f"{best_cause['dimension'].replace('_', ' ')} = {best_cause['segment']}"
        )
        cause_sentence = (
            f"The most credible concentrated cause is {kind_phrase}, which typically represents "
            f"{best_cause['baseline_share_pct']:.1f}% of baseline revenue but fell {best_cause['segment_pct_change']:.1f}% "
            f"this week — {best_cause['decline_ratio_vs_overall']:.2f}x faster than the overall decline rate — "
            f"accounting for {best_cause['contribution_pct_of_drop']:.1f}% of the total dollar drop."
        )
        if classification in ("volume-driven", "compounding"):
            recommendation = (
                f"Recommendation: investigate demand or fulfillment conditions specific to "
                f"{kind_phrase} during this week (e.g. stockouts, seller availability, regional "
                f"disruptions) rather than broad platform-wide factors."
            )
        else:
            recommendation = (
                f"Recommendation: review pricing, discounting, or promotional activity specific to "
                f"{kind_phrase} during this week."
            )

    body = (
        f"{header}\n\n"
        f"ROOT CAUSE:\n{classification_sentence} {cause_sentence}\n\n"
        f"{recommendation}"
    )
    return body


def polish_memo_with_claude(raw_memo: str) -> str:
    """
    OPTIONAL: sends the deterministic memo through the Claude API to
    rewrite it in a sharper, more natural analyst tone, WITHOUT changing
    any numbers or facts — prose polish only.

    Requires:
        pip install anthropic
        export ANTHROPIC_API_KEY="your-key-here"   (see setup notes below)
    """
    try:
        import anthropic
    except ImportError:
        return "[polish_memo_with_claude skipped: run 'pip install anthropic' to enable this feature]"

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return "[polish_memo_with_claude skipped: ANTHROPIC_API_KEY environment variable not set]"

    client = anthropic.Anthropic(api_key=api_key)

    response = client.messages.create(
        model="claude-sonnet-5",
        max_tokens=500,
        system=(
            "You are a business analyst editor. Rewrite the following memo in a sharper, "
            "more natural analyst tone. Do NOT change, add, or remove any numbers, "
            "percentages, dates, or factual claims — only improve sentence flow and word "
            "choice. Keep the same overall structure (header, root cause, recommendation). "
            "Return only the rewritten memo text, nothing else."
        ),
        messages=[{"role": "user", "content": raw_memo}],
    )
    return response.content[0].text


def main():
    weekly = pd.read_parquet(WEEKLY_PATH)
    merged = pd.read_parquet(MERGED_PATH)

    anomalies = detect_revenue_anomalies(weekly)
    flagged = anomalies[anomalies["severity"].isin(["moderate_anomaly", "severe_anomaly"])]

    if flagged.empty:
        print("No anomaly weeks to generate memos for.")
        return

    use_claude = os.environ.get("ANTHROPIC_API_KEY") is not None
    if not use_claude:
        print("(ANTHROPIC_API_KEY not set — skipping Claude-polished memos, showing raw memos only)\n")

    for _, arow in flagged.iterrows():
        week_start = arow["week_start"]

        decomp = decompose_revenue_drop(weekly, week_start)
        if not decomp.get("has_baseline", False):
            continue

        drilldown_results = run_full_drilldown(merged, week_start)
        combo_result = find_combination_effects(
            merged, week_start, "product_category_name_english", "payment_type_mode"
        )

        raw_memo = generate_memo(arow, decomp, drilldown_results, combo_result)

        print("=" * 80)
        print("RAW (deterministic) MEMO:")
        print("=" * 80)
        print(raw_memo)

        if use_claude:
            polished = polish_memo_with_claude(raw_memo)
            print("\n" + "=" * 80)
            print("CLAUDE-POLISHED MEMO:")
            print("=" * 80)
            print(polished)

        print("\n")


if __name__ == "__main__":
    main()
