"""
Tests for src/build_dataset.py — specifically aggregate_payments(), the
function responsible for preventing the fan-out bug documented in Day 1
(joining raw, un-aggregated payments would duplicate order_item rows).
"""
import pandas as pd
from src.build_dataset import aggregate_payments


def test_aggregate_payments_produces_one_row_per_order():
    payments = pd.DataFrame({
        "order_id": ["A", "A", "B", "C", "C", "C"],
        "payment_value": [50.0, 25.0, 100.0, 10.0, 10.0, 5.0],
        "payment_type": ["credit_card", "voucher", "boleto", "credit_card", "credit_card", "voucher"],
        "payment_installments": [1, 1, 1, 3, 2, 1],
    })

    result = aggregate_payments(payments)

    # This is the core guarantee: exactly one row per order_id, no matter
    # how many payment rows that order originally had.
    assert result["order_id"].is_unique
    assert len(result) == payments["order_id"].nunique()


def test_aggregate_payments_sums_value_correctly():
    payments = pd.DataFrame({
        "order_id": ["A", "A", "C", "C", "C"],
        "payment_value": [50.0, 25.0, 10.0, 10.0, 5.0],
        "payment_type": ["credit_card", "voucher", "credit_card", "credit_card", "voucher"],
        "payment_installments": [1, 1, 3, 2, 1],
    })

    result = aggregate_payments(payments)

    a_row = result[result["order_id"] == "A"].iloc[0]
    assert a_row["payment_value_total"] == 75.0

    c_row = result[result["order_id"] == "C"].iloc[0]
    assert c_row["payment_value_total"] == 25.0
    # 2 of 3 rows are credit_card -> majority vote should pick it
    assert c_row["payment_type_mode"] == "credit_card"
    assert c_row["payment_installments_max"] == 3
