"""Part A — Data Foundation: Clean, Store, Query."""
# Author : Juna P
from pathlib import Path
import json
import sqlite3
import pandas as pd

ROOT = Path(__file__).resolve().parent


def run():
    raw = ROOT / 'data/raw'
    clean = ROOT / 'data/clean'
    out = ROOT / '.reports'  
    clean.mkdir(parents=True, exist_ok=True)
    out.mkdir(exist_ok=True)
    tables = {name: pd.read_csv(raw / f'{name}.csv') for name in ('customers', 'products', 'orders')}
    # Audit the raw tables before cleaning.
    audit = []
    for name, frame in tables.items():
        for col in frame:
            audit.append(dict(table=name, rows=len(frame), columns=len(frame.columns), column=col,
                              dtype=str(frame[col].dtype), missing=int(frame[col].isna().sum()),
                              missing_pct=float(frame[col].isna().mean() * 100)))
    pd.DataFrame(audit).to_csv(out / 'raw_audit.csv', index=False)
    customers, products, orders = (tables[name].copy() for name in ('customers', 'products', 'orders'))
    metrics = dict(raw_orders=len(orders), exact_duplicates=int(orders.duplicated().sum()),
                   pending=int(orders.status.eq('pending').sum()), unique_orders=int(orders.order_id.nunique()),
                   nonnumeric_amounts=int(pd.to_numeric(orders.order_amount, errors='coerce').isna().sum()),
                   negative_delivery=int(orders.delivery_minutes.lt(0).sum()), blank_delivery=int(orders.delivery_minutes.isna().sum()),
                   blank_age=int(customers.age.isna().sum()), invalid_age=int((customers.age.notna() & ~customers.age.between(10, 100)).sum()),
                   raw_cities=int(customers.city.nunique()), raw_genders=int(customers.gender.nunique()), blank_gender=int(customers.gender.isna().sum()))
    changes = {}
    city = customers.city.str.strip().str.title()
    changes['city_values_standardised'] = int(customers.city.ne(city).sum())
    customers.city = city
    gender = customers.gender.str.strip().str.lower().replace({'m': 'male', 'f': 'female'}).fillna('unknown')
    changes['gender_values_standardised'] = int(customers.gender.fillna('').ne(gender).sum())
    customers.gender = gender
    customers.loc[~customers.age.between(10, 100), 'age'] = float('nan')
    changes['ages_imputed'] = int(customers.age.isna().sum())
    customers.age = customers.age.fillna(customers.groupby('city').age.transform('median'))
    customers.signup_date = pd.to_datetime(customers.signup_date)
    orders.order_ts = pd.to_datetime(orders.order_ts)
    orders = orders.drop_duplicates()
    changes['exact_duplicates_removed'] = metrics['exact_duplicates']
    before = len(orders)
    orders = orders.sort_values('order_ts', kind='stable').drop_duplicates('order_id', keep='last').copy()
    changes['older_order_records_removed'] = before - len(orders)
    orders.order_amount = pd.to_numeric(orders.order_amount, errors='coerce')
    changes['amounts_reconstructed'] = int(orders.order_amount.isna().sum())
    prices = orders.product_id.map(products.set_index('product_id').price_inr)
    orders.order_amount = orders.order_amount.fillna((prices * orders.quantity * (1 - orders.discount_pct / 100)).round(2))
    changes['negative_delivery_cleared'] = int(orders.delivery_minutes.lt(0).sum())
    orders.loc[orders.delivery_minutes.lt(0), 'delivery_minutes'] = float('nan')
    completed = orders.status.eq('completed')
    changes['completed_delivery_imputed'] = int(orders.loc[completed, 'delivery_minutes'].isna().sum())
    categories = orders.product_id.map(products.set_index('product_id').category)
    medians = orders.loc[completed].groupby(categories[completed]).delivery_minutes.transform('median')
    orders.loc[completed, 'delivery_minutes'] = orders.loc[completed, 'delivery_minutes'].fillna(medians)
    orders.loc[~completed, 'delivery_minutes'] = float('nan')
    with sqlite3.connect(ROOT / 'data/instamart.db') as db:
        for name, frame in [('customers', customers), ('products', products), ('orders', orders)]:
            frame.to_csv(clean / f'{name}.csv', index=False)
            frame.to_sql(name, db, if_exists='replace', index=False)
        queries = (ROOT / 'sql/queries.sql').read_text(encoding='utf-8').split(';')
        results = {}
        for i, query in enumerate(q for q in queries if q.strip()):
            result = pd.read_sql_query(query, db)
            result.to_csv(out / f'q{i+1}.csv', index=False)
            results[f'q{i+1}'] = result
    completed_orders = (orders.loc[completed]
        .merge(customers[['customer_id', 'city']], on='customer_id')
        .merge(products[['product_id', 'category']], on='product_id'))
    pivot = pd.pivot_table(completed_orders, values='order_amount', index='city', columns='category', aggfunc='sum', margins=True)
    pivot.to_csv(out / 'revenue_pivot.csv')
    assert abs(pivot.loc['All', 'All'] - results['q2'].revenue.sum()) < .001
    completed_orders.groupby('city').agg(sum=('order_amount', 'sum'), mean=('order_amount', 'mean'), count=('order_amount', 'count')).to_csv(out / 'city_summary.csv')
    amount = completed_orders.order_amount
    q1, q3 = amount.quantile([.25, .75])
    low, high = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
    corr = completed_orders[['order_amount', 'discount_pct', 'quantity', 'delivery_minutes']].corr()
    corr.to_csv(out / 'correlations.csv')
    stats = dict(mean=float(amount.mean()), median=float(amount.median()), skewness=float(amount.skew()),
                 q1=float(q1), q2=float(amount.median()), q3=float(q3), lower_fence=float(low), upper_fence=float(high),
                 below_fence=int(amount.lt(low).sum()), above_fence=int(amount.gt(high).sum()))
    report = dict(audit=metrics, changes=changes, statistics=stats,
                  clean=dict(customers=len(customers), products=len(products), orders=len(orders), completed=int(completed.sum()),
                             cancelled=int((~completed).sum()), revenue=float(amount.sum())),
                  city_counts={k: int(v) for k, v in customers.city.value_counts().items()},
                  gender_counts={k: int(v) for k, v in customers.gender.value_counts().items()},
                  sql_rows={k: len(v) for k, v in results.items()})
    (out / 'part_a_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    run()
