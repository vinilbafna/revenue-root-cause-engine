# Revenue Root-Cause Diagnostic Engine

## The Problem

Revenue dropped this week. Was it real, or noise? And if it's real — which product category, which region, which payment channel is responsible?

Every e-commerce or marketplace business runs into this moment. A dashboard shows a dip, someone asks "what happened," and the honest answer is usually a scramble: an analyst pulls last week's numbers, slices them five different ways, and either finds a plausible-sounding culprit or runs out of time and guesses. The result is often worse than not answering at all — a segment gets blamed for a decline it didn't cause, a team gets pulled into a fire drill over noise, or a genuine broad-based problem gets misdiagnosed as a "category X issue" because that's the story that was easiest to tell in a Monday morning meeting.

This project exists to replace that scramble with a repeatable, honest process: automatically detect which weeks are actually anomalous, break the revenue change into exactly how much came from fewer orders versus lower order values, and only attribute the drop to a specific segment when the data actually supports it — otherwise say so plainly.

## What This Does

- **Flags anomalous weeks** — automatically identifies weeks where revenue meaningfully deviates from recent normal, instead of relying on someone noticing a dip by eye
- **Explains the "how"** — splits every revenue drop into exactly how much was caused by fewer orders (volume) versus lower average order value (pricing/mix), with no unexplained gap between the two
- **Explains the "where"** — checks whether any single product category, customer state, seller state, or payment method (or combination of these) is disproportionately responsible for the decline
- **Refuses to guess** — if no segment is meaningfully more responsible than any other, it reports the decline as broad-based rather than forcing a scapegoat narrative
- **Writes the memo for you** — generates a plain-English analyst summary for each anomaly week, ready to drop into a report or Slack update

## How It Works

1. **Anomaly detection** — weekly revenue is compared against a rolling baseline; weeks that deviate meaningfully from that baseline are flagged for investigation.
2. **Volume / AOV decomposition** — each flagged week's revenue change is split into a volume effect and an average-order-value effect that sum exactly to the total change, so there's never an "unexplained residual" hiding in the math.
3. **Multi-dimension drilldown with a credibility filter** — every relevant segment (and combinations of segments) is checked to see whether it's disproportionately responsible for the decline. A segment is only flagged as a cause if it clears an explicit credibility bar (meaningful baseline share, a decline rate well above the overall average, and a real dollar impact) — this stops the tool from blaming small, noisy segments just because their percentage swings look dramatic.
4. **Memo generation** — the detection, decomposition, and drilldown results are compiled into a written memo per anomaly week, with an optional Claude API pass to polish the language for a more natural, analyst-style tone.

## Live Demo

🔗 https://revenue-root-cause-engine.streamlit.app/

## Key Design Decisions

**Purchase date, not delivery date.** Revenue is anchored to the date the order was placed, not when it was delivered. Delivery timestamps are downstream of logistics — carrier delays, warehouse backlogs, regional shipping differences — and using them would blur exactly when a demand or pricing shift actually happened. Purchase date is the closest available signal to "when did customer behavior actually change."

**An 8-week baseline window.** Long enough to smooth out normal week-to-week noise and establish a stable "normal," short enough to stay responsive to real, recent shifts in the business (a new pricing strategy, a seasonal shift, etc.). A shorter window overreacts to single bad weeks; a much longer window would mask real trend changes by diluting them into a stale average.

**Geolocation and review data were excluded on purpose.** Both are available in the underlying dataset, but neither answers the question this tool is built to answer. Reviews are submitted after a purchase and lag the revenue event they'd be explaining; geolocation data adds significant complexity without directly explaining *why* a given week's revenue moved. Including either would have inflated scope without improving the diagnosis.

**There is no returns/refunds handling — and that's stated here, not hidden.** The dataset does not provide returns data at a granularity that cleanly ties back to the same order-week structure used for revenue analysis. Rather than approximate this and risk quietly misattributing a decline to the wrong cause, the tool omits returns handling entirely and this is called out explicitly as a limitation below.

## Tech Stack

- **Python / pandas** — data processing and the decomposition/drilldown logic
- **Streamlit** — interactive web app and deployment
- **pytest** — unit tests covering aggregation correctness, exact decomposition math, and the credibility filter
- **Claude API (optional)** — polishes the generated analyst memo into more natural language when an API key is configured

## Dataset

[Olist Brazilian E-Commerce Public Dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) — ~100k real orders placed on the Olist marketplace between 2016 and 2018, including order items, payments, customers, and sellers.

## Limitations

- **Historical, not live.** The dataset covers a fixed 2016–2018 window; this is a diagnostic engine demonstrated on historical data, not a live production pipeline connected to a real-time revenue feed.
- **No returns/refunds modeling.** As noted above, this is intentionally out of scope rather than silently approximated.
- **No external event context.** The tool doesn't automatically cross-reference marketing calendars, promotions, or external market events — a "broad-based decline" result is a strong signal to *go investigate* those manually, not a final answer on its own.
- **Credibility bar thresholds are heuristic.** The specific cutoffs used to decide whether a segment "meaningfully" drove a decline were chosen to be sensible and defensible, not statistically optimized against a larger corpus of labeled anomalies.
- **Memo polishing requires an API key.** Without a configured Claude API key, the app falls back to a clearly-formatted template memo rather than failing.

## Run Locally

```bash
# clone the repo
git clone https://github.com/<your-username>/revenue-root-cause-engine.git
cd revenue-root-cause-engine

# set up a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# install dependencies
pip install -r requirements.txt

# (optional) install dev dependencies to run tests
pip install -r requirements-dev.txt
pytest tests/ -v

# run the app
streamlit run app/main.py   # adjust path if your entry point differs
```

To enable Claude-polished memos locally, create `.streamlit/secrets.toml` (already gitignored) with:

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
