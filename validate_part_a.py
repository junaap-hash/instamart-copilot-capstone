"""Validate regenerated artifacts against the assignment's exact references."""
import json
import sqlite3
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent


def validate():
    c = pd.read_csv(ROOT / 'data/clean/customers.csv')
    o = pd.read_csv(ROOT / 'data/clean/orders.csv')
    report = json.loads((ROOT / '.reports/part_a_report.json').read_text())
    assert report['audit'] == dict(raw_orders=13084, exact_duplicates=119, pending=656,
        unique_orders=12309, nonnumeric_amounts=120, negative_delivery=106, blank_delivery=2541,
        blank_age=61, invalid_age=6, raw_cities=24, raw_genders=10, blank_gender=30)
    assert len(c) == 1200
    assert c.city.value_counts().to_dict() == dict(Pune=227, Delhi=210, Bengaluru=197, Chennai=191, Hyderabad=191, Mumbai=184)
    assert c.gender.value_counts().to_dict() == dict(female=610, male=560, unknown=30)
    assert c.age.notna().all() and c.age.between(18, 60).all()
    assert len(o) == 12309 and o.order_id.is_unique
    assert o.status.value_counts().to_dict() == dict(completed=11110, cancelled=1199)
    assert o.order_amount.notna().all()
    assert o.loc[o.status.eq('completed'), 'delivery_minutes'].notna().all()
    assert o.loc[o.status.eq('cancelled'), 'delivery_minutes'].isna().all()
    assert abs(o.loc[o.status.eq('completed'), 'order_amount'].sum() - 2998668.80) < .001
    assert o.customer_id.isin(c.customer_id).all()
    products = pd.read_csv(ROOT / 'data/clean/products.csv')
    assert len(products) == 40 and o.product_id.isin(products.product_id).all()
    with sqlite3.connect(ROOT / 'data/instamart.db') as db:
        frames = [pd.read_sql_query(q, db) for q in (ROOT / 'sql/queries.sql').read_text().split(';') if q.strip()]
        assert [len(f) for f in frames] == [30, 30, 11, 25, 550, 1175, 183]
        for table, expected in [('customers', 1200), ('products', 40), ('orders', 12309)]:
            assert db.execute(f'SELECT COUNT(*) FROM {table}').fetchone()[0] == expected
    revenue = frames[1]
    assert abs(revenue.revenue.sum() - 2998668.80) < .001
    largest = revenue.loc[revenue.revenue.idxmax()]
    assert (largest.city, largest.category) == ('Bengaluru', 'Electronics')
    assert abs(largest.revenue - 233909.45) < .001
    assert round(frames[4].average_spend.iloc[0], 2) == 2552.06
    ranked = frames[5]
    assert ranked.spend_rank.ne(ranked.row_number).sum() == 8
    expected_top = dict(Bengaluru=('C0082', 12856.20), Chennai=('C0930', 10151.40),
        Delhi=('C0291', 13742.60), Hyderabad=('C0743', 12730.05), Mumbai=('C0693', 14766.55), Pune=('C0426', 10377.10))
    for row in ranked.loc[ranked.spend_rank.eq(1)].itertuples():
        cid, spend = expected_top[row.city]
        assert row.customer_id == cid and abs(row.total_spend - spend) < .001
    assert abs(frames[6].cumulative_revenue.iloc[-1] - 2998668.80) < .001
    assert report['statistics']['above_fence'] == 897 and report['statistics']['below_fence'] == 0
    print('ALL CHECKPOINTS PASSED')


if __name__ == '__main__':
    validate()
