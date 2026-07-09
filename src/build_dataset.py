"""
build_dataset.py

Builds the analysis-ready order-item-level dataset for the Revenue Root
Cause Diagnostic Engine.

Excludes olist_geolocation_dataset.csv (fan-out risk, no analytical value)
and olist_order_reviews_dataset.csv (out of scope for revenue causation).

Join grain: one row per order ITEM (order_items is the base table).
Output: data/processed/merged_orders.parquet

Run from project root with the venv active:
    python -m src.build_dataset
"""

from pathlib import Path
import pandas as pd

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"

ID_COLUMNS = [
    "customer_id", "customer_unique_id", "order_id", "product_id",
    "seller_id", "customer_zip_code_prefix", "seller_zip_code_prefix",
]


def _dtype_map_for(df_columns) -> dict:
    return {col: "string" for col in ID_COLUMNS if col in df_columns}


# ---- STEP 1: Load all 7 CSVs ----
def load_raw_tables() -> dict[str, pd.DataFrame]:
    def read(filename, date_cols=None):
        path = RAW_DIR / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing raw file: {path}")
        header = pd.read_csv(path, nrows=0).columns
        return pd.read_csv(
            path, dtype=_dtype_map_for(header),
            parse_dates=date_cols, low_memory=False,
        )

    return {
        "order_items": read("olist_order_items_dataset.csv", date_cols=["shipping_limit_date"]),
        "products": read("olist_products_dataset.csv"),
        "category_translation": read("product_category_name_translation.csv"),
        "sellers": read("olist_sellers_dataset.csv"),
        "orders": read("olist_orders_dataset.csv", date_cols=[
            "order_purchase_timestamp", "order_approved_at",
            "order_delivered_carrier_date", "order_delivered_customer_date",
            "order_estimated_delivery_date",
        ]),
        "customers": read("olist_customers_dataset.csv"),
        "order_payments": read("olist_order_payments_dataset.csv"),
    }


# ---- STEP 2: Aggregate payments to order_id level FIRST ----
def aggregate_payments(order_payments: pd.DataFrame) -> pd.DataFrame:
    """
    MUST happen before any join. order_payments has MULTIPLE rows per
    order_id (e.g. split payment across card + voucher). Our base table,
    order_items, is at order-ITEM grain. Joining raw multi-row payments
    directly onto order_items would fan-out: every extra payment row for
    an order multiplies every item row of that order, silently inflating
    row counts and any revenue sum computed afterward. So we collapse
    payments to exactly one row per order_id FIRST, then join that single
    summarized row onto each item row.
    """
    return (
        order_payments.groupby("order_id")
        .agg(
            payment_value_total=("payment_value", "sum"),
            payment_type_mode=("payment_type", lambda s: s.mode().iat[0] if not s.mode().empty else None),
            payment_installments_max=("payment_installments", "max"),
        )
        .reset_index()
    )


# ---- STEP 3: Join in exact sequence ----
def build_merged_dataset(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    df = tables["order_items"].copy()
    starting_rows = len(df)

    df = df.merge(tables["products"], on="product_id", how="left")               # -> products
    df = df.merge(tables["category_translation"], on="product_category_name", how="left")  # -> category translation
    df = df.merge(tables["sellers"], on="seller_id", how="left")                 # -> sellers
    df = df.merge(tables["orders"], on="order_id", how="left")                   # -> orders
    df = df.merge(tables["customers"], on="customer_id", how="left")             # -> customers
    payments_agg = aggregate_payments(tables["order_payments"])
    df = df.merge(payments_agg, on="order_id", how="left")                       # -> aggregated payments

    assert len(df) == starting_rows, (
        f"Row count changed during joins: {starting_rows} -> {len(df)}. "
        f"A join fanned out — investigate before trusting revenue numbers."
    )
    return df


# ---- STEP 4: Derived column ----
def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    df["order_revenue_item"] = df["price"] + df["freight_value"]
    return df


# ---- STEP 5: Drop unnecessary columns ----
def drop_unnecessary_columns(df: pd.DataFrame) -> pd.DataFrame:
    columns_to_drop = [
        "product_name_lenght", "product_description_lenght", "product_photos_qty",
        "product_weight_g", "product_length_cm", "product_height_cm", "product_width_cm",
        "product_category_name", "customer_zip_code_prefix", "seller_zip_code_prefix",
        "customer_city", "seller_city",
    ]
    existing = [c for c in columns_to_drop if c in df.columns]
    return df.drop(columns=existing)


# ---- STEP 7: Sanity check ----
def sanity_check_revenue(df: pd.DataFrame, order_payments_raw: pd.DataFrame) -> None:
    derived_total = df["order_revenue_item"].sum()
    raw_payments_total = order_payments_raw["payment_value"].sum()
    diff = derived_total - raw_payments_total
    pct_diff = (diff / raw_payments_total) * 100 if raw_payments_total else float("nan")

    print("\n--- Revenue Sanity Check ---")
    print(f"Sum of order_revenue_item (price + freight, item-level): {derived_total:,.2f}")
    print(f"Sum of raw payment_value (order_payments.csv):           {raw_payments_total:,.2f}")
    print(f"Difference:                                              {diff:,.2f} ({pct_diff:.3f}%)")
    print("(Small differences are expected — e.g. vouchers/discounts. A LARGE")
    print(" difference or exact multiple would indicate row duplication.)")


def main():
    print("Loading raw tables...")
    tables = load_raw_tables()

    print("Joining tables...")
    df = build_merged_dataset(tables)

    print("Adding derived columns...")
    df = add_derived_columns(df)

    print("Dropping unnecessary columns...")
    df = drop_unnecessary_columns(df)

    sanity_check_revenue(df, tables["order_payments"])

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PROCESSED_DIR / "merged_orders.parquet"
    df.to_parquet(output_path, index=False)
    print(f"\nSaved merged dataset to: {output_path}")

    print("\n--- Final DataFrame shape ---")
    print(df.shape)

    print("\n--- Final DataFrame info ---")
    df.info()


if __name__ == "__main__":
    main()
