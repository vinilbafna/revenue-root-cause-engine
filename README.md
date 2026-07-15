# Revenue Root-Cause Diagnostic Engine

## The Problem

Revenue dropped this week. Was it real, or noise? And if it's real — which product category, which region, which payment channel is responsible?

Every e-commerce or marketplace business runs into this moment. A dashboard shows a dip, someone asks "what happened," and the honest answer is usually a scramble: an analyst pulls last week's numbers, slices them five different ways, and either finds a plausible-sounding culprit or runs out of time and guesses. The result is often worse than not answering at all — a segment gets blamed for a decline it didn't cause, or a genuine broad-based problem gets misdiagnosed as a "category X issue" because that's the easiest story to tell in a Monday morning meeting.

This project replaces that scramble with a repeatable, honest process: automatically detect which weeks are actually anomalous, break the revenue change into exactly how much came from fewer orders versus lower order values, and only attribute the drop to a specific segment when the data actually supports it — otherwise, say so plainly.

**[Live demo →](https://revenue-root-cause-engine.streamlit.app)**
*(Hosted on Streamlit's free tier — if it shows a "waking up" screen, give it ~30–60 seconds; it sleeps after 12 hours of inactivity.)*

## What It Does

1. **Anomaly detection** — flags weeks where revenue falls statistically below its trailing 8-week baseline (z-score based), excluding the week itself from its own baseline so a genuine anomaly doesn't dampen its own signal
2. **Volume vs. AOV decomposition** — an exact split of *why* revenue changed (fewer orders vs. lower average order value), verified to sum with zero residual against a ground-truth synthetic scenario
3. **Segment drilldown** — checks whether any category, customer state, seller state, or payment method (or combination) is disproportionately responsible, filtering out segments that are statistically dramatic but economically insignificant (e.g. a category down -100% but representing under 1% of revenue)
4. **Memo generation** — produces a plain-English report per anomaly week, explicitly stating "broad-based decline, no concentrated cause" rather than fabricating an explanation when the data doesn't support one, with an optional Claude API pass to polish the language

## How It Works

Weekly revenue is compared against a rolling 8-week baseline; weeks that deviate meaningfully are flagged. Each flagged week's revenue change is decomposed into a volume effect and an AOV effect that sum exactly to the total change — no unexplained residual hiding in the math. Every relevant segment (and combinations of segments) is then checked against a two-condition credibility filter, and only flagged as a cause if it clears both bars. Finally, the detection, decomposition, and drilldown results are compiled into a memo per anomaly week.

## Key Engineering Decisions

- **Purchase date, not delivery date** — revenue is anchored to when the order was placed, not delivered. Delivery timestamps are downstream of logistics (carrier delays, warehouse backlogs) and would blur exactly when a real demand or pricing shift happened.
- **An 8-week baseline window** — long enough to smooth normal week-to-week noise into a stable "normal," short enough to stay responsive to real recent shifts. Shorter overreacts to one bad week; much longer dilutes genuine trend changes into a stale average.
- **Payments aggregated before joining** — prevents row fan-out from Olist's multi-row payment records, verified with a hard row-count assertion rather than assumed.
- **Two-condition credibility filter** (≥3% baseline share AND >1.3x the overall decline rate, real dollar decline) — prevents small, noisy segments with dramatic percentage swings from being mistaken for real root causes.
- **Exact additive decomposition** — `volume_effect + aov_effect` sums exactly to the true revenue change, unit-tested against a ground-truth synthetic scenario, not just checked for internal self-consistency.
- **Geolocation and review data excluded on purpose** — both exist in the underlying dataset but don't answer *why* a given week's revenue moved. Reviews are submitted after purchase and lag the event they'd explain; geolocation adds complexity without improving the diagnosis.
- **No returns/refunds handling — stated, not hidden** — the dataset has no returns table; order cancellation status is the full extent of "returns handling" possible with this data, and that's called out explicitly in Limitations below rather than silently approximated.

## Bugs Found and Fixed During Development

- A `revenue_baseline` calculation that was internally self-consistent but not ground-truth correct — the mean of a product isn't the product of the means. Caught by independently verifying output against a hand-computed `groupby()`, not by trusting that the code ran without errors.
- A test that still passed even after that bug was deliberately reintroduced, because it only checked self-consistency rather than a known correct value — fixed by adding a ground-truth assertion, then re-verified the test genuinely fails on broken code and passes on correct code.
- A memo-generation language bug that described a -35.9% AOV change as "comparatively stable" — caught by reading the generated memo text aloud, not just confirming it printed.

## Tech Stack

Python · pandas · Streamlit · Plotly · pytest · Claude API (optional, for memo polishing)

## Dataset

[Olist Brazilian E-Commerce Public Dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) (Kaggle) — ~100k real orders placed on the Olist marketplace between 2016 and 2018, including order items, payments, customers, and sellers.

## Limitations (stated honestly)

- Anomaly detection thresholds (8-week window, z-score cutoff, 3%/1.3x credibility bar) are reasonable defaults, not backtested against ground-truth labeled events.
- No returns/refunds modeling — order cancellation status is the closest available proxy, and this is a real gap, not a silently-assumed non-issue.
- No external event context — the tool doesn't cross-reference marketing calendars or market events; a "broad-based decline" result is a signal to go investigate those manually, not a final answer on its own.
- No CI pipeline yet — tests are run manually with `pytest tests/ -v`, not automatically on push.
- The "catches a real bug" property has been directly proven for the decomposition module (by deliberately reintroducing a known bug and confirming the test fails); the other three test files follow the same pattern but haven't each been individually stress-tested this way.
- Historical, not live — this is a diagnostic engine demonstrated on a fixed 2016–2018 dataset, not connected to a real-time revenue feed.

## Run Locally

```bash
git clone https://github.com/vinilbafna/revenue-root-cause-engine.git
cd revenue-root-cause-engine

python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r requirements.txt
pip install -r requirements-dev.txt   # optional, for running tests

pytest tests/ -v                       # optional
streamlit run app/app.py
```

To enable Claude-polished memos locally, create `.streamlit/secrets.toml` (already gitignored):

```toml
ANTHROPIC_API_KEY = "your-key-here"
```

## Project Structure

```
├── app/                        # Streamlit app entry point and UI
├── src/                        # Core logic
│   ├── build_dataset.py        # Raw data ingestion and aggregation
│   ├── anomaly_detector.py     # Rolling-baseline anomaly detection
│   ├── decompose_drop.py       # Exact volume / AOV decomposition
│   ├── drilldown.py            # Multi-dimension credibility-filtered drilldown
│   └── memo.py                 # Analyst memo generation
├── tests/                      # pytest unit tests
├── data/                       # Raw (gitignored) and processed data
├── requirements.txt
├── requirements-dev.txt
└── runtime.txt                 # Pinned Python version for Streamlit Cloud
```
