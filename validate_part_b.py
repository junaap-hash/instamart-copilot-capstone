"""Check Part B artifacts, model predictions, references and cutoff isolation."""
import json
import joblib
import numpy as np
import pandas as pd

from part_b import (OUT, REQUIRED_OUT, NUMERIC, CATEGORICAL, FEATURES, CUTOFF, load_data,
                    build_rfm, build_lapse_dataset, split_lapse_data, metrics)


def validate():
    r = json.loads((OUT / 'part_b_report.json').read_text())
    orders, customers, products = load_data()
    segments = pd.read_csv(REQUIRED_OUT / 'customer_segments.csv').set_index('customer_id')
    assert segments.columns.tolist() == ['recency', 'frequency', 'monetary', 'segment_id', 'segment_name', 'is_premium']
    assert segments.index.is_unique and len(segments) == 1175
    pd.testing.assert_frame_equal(segments[['recency', 'frequency', 'monetary']], build_rfm(orders))
    assert (segments.recency.median(), segments.recency.min(), segments.recency.max()) == (19, 0, 180)
    assert abs(segments.frequency.mean() - 9.46) < .01 and segments.frequency.max() == 33
    assert abs(segments.monetary.mean() - 2552.06) < .01
    assert abs(segments.monetary.max() - 14766.55) < .01
    profiles = pd.read_csv(OUT / 'segment_profiles.csv').set_index('segment_id')
    calculated = segments.groupby('segment_id').agg(size=('monetary', 'size'),
        mean_recency=('recency', 'mean'), mean_frequency=('frequency', 'mean'), mean_monetary=('monetary', 'mean'))
    pd.testing.assert_frame_equal(profiles[calculated.columns], calculated)
    assert sorted(profiles['size']) == [138, 182, 419, 436]
    premium = profiles.mean_monetary.idxmax()
    assert np.array_equal(segments.is_premium, segments.segment_id.eq(premium))
    assert abs(profiles.loc[premium, 'mean_monetary'] - 7130) < 100
    assert abs(profiles.loc[premium, 'mean_frequency'] - 6.8) < .2
    frequent = profiles.loc[profiles['size'].eq(419)].iloc[0]
    assert abs(frequent.mean_frequency - 19.4) < .2 and abs(frequent.mean_recency - 8.8) < .2
    comparison = pd.read_csv(OUT / 'cluster_comparison.csv')
    assert comparison.k.tolist() == list(range(2, 9))
    assert np.allclose(comparison.inertia, [1820.8, 1268.2, 743.1, 617.4, 503.2, 408.5, 366.5], rtol=.05)
    assert np.allclose(comparison.silhouette, [.465, .445, .526, .481, .482, .416, .412], atol=.05)
    assert comparison.inertia.diff().dropna().lt(0).all()
    assert r['clustering']['chosen_k'] == 4
    assert np.allclose(r['clustering']['pca_explained_variance_ratio'], [.634, .218], atol=.02)
    assert abs(r['clustering']['pca_total'] - .852) < .02
    dataset, provenance = build_lapse_dataset(orders, customers, products)
    assert len(dataset) == 1156 and dataset.lapsed.sum() == 443
    assert pd.Timestamp(provenance['latest_feature_order_ts']) < CUTOFF
    assert not dataset[FEATURES].isna().any().any()
    x_train, x_test, y_train, y_test = split_lapse_data(dataset)
    assert len(x_train) == 924 and len(x_test) == 232
    assert set(x_train.index).isdisjoint(x_test.index)
    assert y_train.value_counts().to_dict() == {0: 570, 1: 354}
    assert y_test.value_counts().to_dict() == {0: 143, 1: 89}
    saved_cohort = pd.read_csv(OUT / 'lapse_cohort.csv').set_index('customer_id')
    pd.testing.assert_frame_equal(saved_cohort[dataset.columns], dataset)
    assert set(saved_cohort.index[saved_cohort.split.eq('train')]) == set(x_train.index)
    # Removing all September rows must leave every feature unchanged.
    no_september, _ = build_lapse_dataset(orders.loc[orders.order_ts.lt(CUTOFF)], customers, products)
    pd.testing.assert_frame_equal(dataset[FEATURES], no_september[FEATURES])
    model = joblib.load(REQUIRED_OUT / 'lapse_model.joblib')
    assert model.feature_names_in_.tolist() == FEATURES
    scaler = model.named_steps['preprocess'].named_transformers_['numeric']
    assert scaler.n_samples_seen_ == len(x_train)
    assert np.allclose(scaler.mean_, x_train[NUMERIC].mean())
    encoder = model.named_steps['preprocess'].named_transformers_['categorical']
    assert encoder.drop == 'first'
    for feature, learned_categories in zip(CATEGORICAL, encoder.categories_):
        assert learned_categories.tolist() == sorted(x_train[feature].unique())
    assert abs(r['preprocessing']['training_frequency_mean'] - x_train.frequency.mean()) < 1e-10
    assert abs(r['preprocessing']['cohort_frequency_mean'] - dataset.frequency.mean()) < 1e-10
    logistic_cv = pd.read_csv(OUT / 'logistic_regression_cv.csv')
    forest_cv = pd.read_csv(OUT / 'random_forest_cv.csv')
    assert sorted(logistic_cv.param_model__C) == [.01, .1, 1, 10, 100]
    assert len(forest_cv) == 9 and set(forest_cv.param_model__max_depth) == {3, 5, 8}
    assert set(forest_cv.param_model__min_samples_leaf) == {5, 10, 20}
    for frame in (logistic_cv, forest_cv):
        assert all(f'split{i}_test_score' in frame for i in range(5))
        assert frame.mean_test_score.notna().all()
    predictions = pd.read_csv(OUT / 'lapse_test_predictions.csv').set_index('customer_id').loc[x_test.index]
    assert np.array_equal(predictions.actually_lapsed, y_test)
    assert np.array_equal(predictions[r['selected_model']], model.predict(x_test))
    assert np.allclose(predictions.selected_lapse_probability, model.predict_proba(x_test)[:, 1], atol=1e-14)
    chosen = max(r['models'], key=lambda name: r['models'][name]['cv_f1'])
    assert chosen == r['selected_model']
    for name in r['models']:
        measured = metrics(y_test, predictions[name])
        assert measured == r['models'][name]['test']
    assert metrics(y_test, model.predict(x_test))['f1'] >= .55
    assert r['baseline'] == metrics(y_test, np.zeros(len(y_test), dtype=int))
    assert abs(r['baseline']['accuracy'] - .616) < .001 and r['baseline']['f1'] == 0
    assert r['oversampling']['balanced_training_counts'] == {'0': 570, '1': 570}
    assert r['oversampling']['duplicated_training_rows'] == 216
    assert r['persistence']['predictions_identical'] and r['persistence']['probabilities_identical']
    importance = pd.read_csv(OUT / 'forest_importances.csv')
    assert abs(importance.importance.sum() - 1) < 1e-10
    assert len(r['forest_top_five']) == 5 and len(r['logistic_three_coefficients']) == 3
    retention = pd.read_csv(REQUIRED_OUT / 'retention_list.csv')
    assert retention.columns.tolist() == ['customer_id', 'lapse_probability', 'actually_lapsed']
    eligible = predictions.loc[predictions.index.isin(segments.index[segments.is_premium])].copy()
    eligible['customer_id'] = eligible.index
    expected = eligible.reset_index(drop=True).sort_values(['selected_lapse_probability', 'customer_id'], ascending=[False, True]).head(10)
    assert retention.customer_id.tolist() == expected.customer_id.tolist()
    assert np.allclose(retention.lapse_probability, expected.selected_lapse_probability)
    assert np.array_equal(retention.actually_lapsed, expected.actually_lapsed)
    assert retention.actually_lapsed.sum() == r['retention']['actually_lapsed']
    print('ALL PART B CHECKPOINTS PASSED')


if __name__ == '__main__':
    validate()
