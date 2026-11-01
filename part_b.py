"""Customer segments and leakage-controlled lapse prediction (Part B)."""
from pathlib import Path
import json
import os
import platform

# Bound native numeric workers before importing numerical libraries.
for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(variable, '1')

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, precision_score,
                             recall_score, f1_score, silhouette_score)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
OUT = ROOT / '.reports'  # Supplementary reports and verification evidence.
REQUIRED_OUT = ROOT / 'outputs'
SNAPSHOT = pd.Timestamp('2025-10-01')
CUTOFF = pd.Timestamp('2025-09-01')
NUMERIC = ['recency', 'frequency', 'monetary', 'average_discount_pct',
           'cancellation_rate', 'average_delivery_minutes', 'age', 'tenure_days']
CATEGORICAL = ['city', 'gender', 'favourite_category']
FEATURES = NUMERIC + CATEGORICAL


def load_data():
    return (pd.read_csv(ROOT / 'data/clean/orders.csv', parse_dates=['order_ts']),
            pd.read_csv(ROOT / 'data/clean/customers.csv', parse_dates=['signup_date']),
            pd.read_csv(ROOT / 'data/clean/products.csv'))


def build_rfm(orders, snapshot=SNAPSHOT):
    """One row per customer with completed history before the snapshot."""
    completed = orders.loc[orders.status.eq('completed') & orders.order_ts.lt(snapshot)]
    rfm = completed.groupby('customer_id').agg(
        last_order_ts=('order_ts', 'max'), frequency=('order_id', 'nunique'),
        monetary=('order_amount', 'sum'))
    rfm['recency'] = (snapshot - rfm.pop('last_order_ts')).dt.days
    return rfm[['recency', 'frequency', 'monetary']]


def compare_clusters(rfm):
    scaled = StandardScaler().fit_transform(rfm)
    rows, models = [], {}
    for k in range(2, 9):
        model = KMeans(n_clusters=k, n_init=10, random_state=42).fit(scaled)
        models[k] = model
        rows.append(dict(k=k, inertia=float(model.inertia_),
                         silhouette=float(silhouette_score(scaled, model.labels_))))
    return scaled, pd.DataFrame(rows), models


def profile_segments(rfm, labels):
    segments = rfm.copy()
    segments['segment_id'] = labels
    profiles = segments.groupby('segment_id').agg(
        size=('monetary', 'size'), mean_recency=('recency', 'mean'),
        mean_frequency=('frequency', 'mean'), mean_monetary=('monetary', 'mean'))
    premium = int(profiles.mean_monetary.idxmax())
    remaining = profiles.drop(index=premium)
    frequent = int(remaining.mean_frequency.idxmax())
    dormant = int(remaining.drop(index=frequent).mean_recency.idxmax())
    names = {premium: 'Premium', frequent: 'Frequent Essentials', dormant: 'Dormant'}
    actions = {premium: 'Offer priority service and tailored high-value retention.',
               frequent: 'Reward repeat purchases with essentials bundles.',
               dormant: 'Test a capped win-back offer and measure response.'}
    for segment_id in profiles.index:
        names.setdefault(int(segment_id), 'Occasional Shoppers')
        actions.setdefault(int(segment_id), 'Encourage a second routine purchase with relevant reminders.')
    profiles['segment_name'] = profiles.index.map(names)
    profiles['recommended_action'] = profiles.index.map(actions)
    segments['segment_name'] = segments.segment_id.map(names)
    segments['is_premium'] = segments.segment_id.eq(premium)
    return segments, profiles, premium


def export_segments(segments, profiles):
    segments.to_csv(REQUIRED_OUT / 'customer_segments.csv', index=True)
    profiles.to_csv(OUT / 'segment_profiles.csv', index=True)


def build_lapse_dataset(orders, customers, products):
    """Use September only for y, never for feature aggregation."""
    prior = orders.loc[orders.order_ts.lt(CUTOFF)].copy()
    assert prior.order_ts.lt(CUTOFF).all()
    completed = prior.loc[prior.status.eq('completed')].copy()
    dataset = build_rfm(prior, CUTOFF)
    other = completed.groupby('customer_id').agg(
        average_discount_pct=('discount_pct', 'mean'),
        average_delivery_minutes=('delivery_minutes', 'mean'))
    final = prior.loc[prior.status.isin(['completed', 'cancelled'])].copy()
    final['cancelled'] = final.status.eq('cancelled')
    other['cancellation_rate'] = final.groupby('customer_id').cancelled.mean()
    completed['category'] = completed.product_id.map(products.set_index('product_id').category)
    # Equal-frequency categories are resolved alphabetically, deterministically.
    other['favourite_category'] = completed.groupby('customer_id').category.agg(lambda s: s.mode().sort_values().iloc[0])
    customer_info = customers.set_index('customer_id')[['city', 'age', 'gender', 'signup_date']].copy()
    customer_info['tenure_days'] = (CUTOFF - customer_info.pop('signup_date')).dt.days
    dataset = dataset.join(other).join(customer_info)
    september_ids = orders.loc[orders.status.eq('completed') &
        orders.order_ts.ge(CUTOFF) & orders.order_ts.lt(SNAPSHOT), 'customer_id']
    dataset['lapsed'] = (~dataset.index.isin(september_ids)).astype(int)
    assert not dataset[FEATURES].isna().any().any()
    assert not any('segment' in column or 'premium' in column for column in FEATURES)
    assert dataset.index.is_unique
    provenance = dict(cutoff=str(CUTOFF), latest_feature_order_ts=str(prior.order_ts.max()),
                      latest_completed_feature_order_ts=str(completed.order_ts.max()),
                      feature_order_count=len(prior), completed_feature_order_count=len(completed),
                      favourite_category_tie_break='alphabetically first mode',
                      feature_columns=FEATURES)
    return dataset, provenance


def split_lapse_data(dataset):
    return train_test_split(dataset[FEATURES], dataset.lapsed, test_size=0.2,
                            stratify=dataset.lapsed, random_state=42)


def make_preprocessor():
    return ColumnTransformer([
        ('numeric', StandardScaler(), NUMERIC),
        ('categorical', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False), CATEGORICAL)])


def tune_models(x_train, y_train):
    specifications = {
        'logistic_regression': (LogisticRegression(max_iter=1000), {'model__C': [.01, .1, 1, 10, 100]}),
        'random_forest': (RandomForestClassifier(n_estimators=100, random_state=42),
                          {'model__max_depth': [3, 5, 8], 'model__min_samples_leaf': [5, 10, 20]})}
    searches = {}
    for name, (model, parameters) in specifications.items():
        pipeline = Pipeline([('preprocess', make_preprocessor()), ('model', model)])
        search = GridSearchCV(pipeline, parameters, cv=5, scoring='f1', n_jobs=1,
                              error_score='raise', return_train_score=False)
        search.fit(x_train, y_train)
        pd.DataFrame(search.cv_results_).to_csv(OUT / f'{name}_cv.csv', index=False)
        searches[name] = search
    return searches


def metrics(y_true, predicted):
    return dict(accuracy=float(accuracy_score(y_true, predicted)),
                precision=float(precision_score(y_true, predicted, zero_division=0)),
                recall=float(recall_score(y_true, predicted, zero_division=0)),
                f1=float(f1_score(y_true, predicted, zero_division=0)),
                confusion_matrix=confusion_matrix(y_true, predicted, labels=[0, 1]).tolist())


def evaluate_models(searches, x_test, y_test):
    return {name: dict(best_parameters=search.best_params_, cv_f1=float(search.best_score_),
                       test=metrics(y_test, search.predict(x_test)))
            for name, search in searches.items()}


def compare_oversampling(best_logistic, x_train, y_train, x_test, y_test):
    """Duplicate only training minority rows; leave test rows untouched."""
    counts = y_train.value_counts()
    positive_ids = y_train.index[y_train.eq(1)]
    extras = np.random.RandomState(42).choice(positive_ids, size=int(counts[0] - counts[1]), replace=True)
    x_balanced = pd.concat([x_train, x_train.loc[extras]])
    y_balanced = pd.concat([y_train, y_train.loc[extras]])
    assert set(x_balanced.index).isdisjoint(x_test.index)
    balanced_model = Pipeline([('preprocess', make_preprocessor()),
        ('model', LogisticRegression(max_iter=1000, C=best_logistic.best_params_['model__C']))])
    balanced_model.fit(x_balanced, y_balanced)
    return dict(original_training_counts={str(k): int(v) for k, v in counts.items()},
                balanced_training_counts={str(k): int(v) for k, v in y_balanced.value_counts().items()},
                duplicated_training_rows=len(extras),
                unbalanced=metrics(y_test, best_logistic.predict(x_test)),
                balanced=metrics(y_test, balanced_model.predict(x_test)))


def interpret_models(searches):
    forest = searches['random_forest'].best_estimator_
    logistic = searches['logistic_regression'].best_estimator_
    importance = pd.DataFrame(dict(feature=forest.named_steps['preprocess'].get_feature_names_out(),
        importance=forest.named_steps['model'].feature_importances_)).sort_values('importance', ascending=False)
    coefficients = pd.DataFrame(dict(feature=logistic.named_steps['preprocess'].get_feature_names_out(),
        coefficient=logistic.named_steps['model'].coef_[0]))
    importance.to_csv(OUT / 'forest_importances.csv', index=False)
    coefficients.to_csv(OUT / 'logistic_coefficients.csv', index=False)
    return importance.head(5).to_dict('records'), coefficients.loc[
        coefficients.feature.isin(['numeric__recency', 'numeric__frequency', 'numeric__monetary'])].to_dict('records')


def persist_model(model, x_test):
    joblib.dump(model, REQUIRED_OUT / 'lapse_model.joblib')
    reloaded = joblib.load(REQUIRED_OUT / 'lapse_model.joblib')
    assert np.array_equal(model.predict(x_test), reloaded.predict(x_test))
    assert np.array_equal(model.predict_proba(x_test), reloaded.predict_proba(x_test))
    return reloaded


def export_retention_list(model, x_test, y_test, segments):
    test = pd.DataFrame(dict(customer_id=x_test.index,
        lapse_probability=model.predict_proba(x_test)[:, 1], actually_lapsed=y_test.to_numpy()))
    test['is_premium'] = test.customer_id.map(segments.is_premium).fillna(False)
    retention = test.loc[test.is_premium].sort_values(
        ['lapse_probability', 'customer_id'], ascending=[False, True]).head(10).drop(columns='is_premium')
    retention.to_csv(REQUIRED_OUT / 'retention_list.csv', index=False)
    return dict(premium_test_customers=int(test.is_premium.sum()), retained_rows=len(retention),
                actually_lapsed=int(retention.actually_lapsed.sum()))


def run():
    OUT.mkdir(exist_ok=True)
    REQUIRED_OUT.mkdir(exist_ok=True)
    orders, customers, products = load_data()
    rfm = build_rfm(orders)
    scaled, cluster_scores, cluster_models = compare_clusters(rfm)
    cluster_scores.to_csv(OUT / 'cluster_comparison.csv', index=False)
    # Four clusters balance the strong elbow/silhouette with distinct business actions.
    segments, profiles, premium = profile_segments(rfm, cluster_models[4].labels_)
    export_segments(segments, profiles)
    pca = PCA(n_components=2).fit(scaled)
    dataset, provenance = build_lapse_dataset(orders, customers, products)
    x_train, x_test, y_train, y_test = split_lapse_data(dataset)
    assert set(x_train.index).isdisjoint(x_test.index)
    dataset['split'] = np.where(dataset.index.isin(x_train.index), 'train', 'test')
    dataset.to_csv(OUT / 'lapse_cohort.csv')
    searches = tune_models(x_train, y_train)
    # Select using training CV alone; only then evaluate the fixed models on test.
    chosen = max(searches, key=lambda name: searches[name].best_score_)
    evaluations = evaluate_models(searches, x_test, y_test)
    assert evaluations[chosen]['test']['f1'] >= .55
    sampling = compare_oversampling(searches['logistic_regression'], x_train, y_train, x_test, y_test)
    importance, coefficients = interpret_models(searches)
    model = persist_model(searches[chosen].best_estimator_, x_test)
    retention = export_retention_list(model, x_test, y_test, segments)
    predictions = pd.DataFrame({'customer_id': x_test.index, 'actually_lapsed': y_test.to_numpy()})
    for name, search in searches.items():
        predictions[name] = search.predict(x_test)
    predictions['selected_lapse_probability'] = model.predict_proba(x_test)[:, 1]
    predictions.to_csv(OUT / 'lapse_test_predictions.csv', index=False)
    scaler = model.named_steps['preprocess'].named_transformers_['numeric']
    report = dict(versions=dict(python=platform.python_version(), sklearn=sklearn.__version__, pandas=pd.__version__),
        rfm=dict(customers=len(rfm), recency_median=float(rfm.recency.median()), recency_min=int(rfm.recency.min()),
                 recency_max=int(rfm.recency.max()), frequency_mean=float(rfm.frequency.mean()),
                 frequency_max=int(rfm.frequency.max()), monetary_mean=float(rfm.monetary.mean()),
                 monetary_max=float(rfm.monetary.max())),
        clustering=dict(chosen_k=4, premium_segment_id=premium,
            pca_explained_variance_ratio=pca.explained_variance_ratio_.tolist(),
            pca_total=float(pca.explained_variance_ratio_.sum())),
        cohort=dict(customers=len(dataset), lapsed=int(dataset.lapsed.sum()), lapse_pct=float(dataset.lapsed.mean()*100),
                    training_rows=len(x_train), test_rows=len(x_test),
                    train_class_counts={str(k): int(v) for k, v in y_train.value_counts().items()},
                    test_class_counts={str(k): int(v) for k, v in y_test.value_counts().items()}),
        provenance=provenance,
        preprocessing=dict(training_frequency_mean=float(scaler.mean_[NUMERIC.index('frequency')]),
            cohort_frequency_mean=float(dataset.frequency.mean()), training_rows_seen=int(scaler.n_samples_seen_),
            no_missing_features=True, encoded_feature_count=len(model.named_steps['preprocess'].get_feature_names_out())),
        baseline=metrics(y_test, np.zeros(len(y_test), dtype=int)), models=evaluations,
        selected_model=chosen, selection_rule='highest training five-fold cross-validated F1',
        oversampling=sampling, forest_top_five=importance, logistic_three_coefficients=coefficients,
        persistence=dict(predictions_identical=True, probabilities_identical=True), retention=retention)
    (OUT / 'part_b_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    run()
