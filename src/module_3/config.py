PREPROCESSOR_CONFIG = {
    'datetime_col': 'created_at',
    'user_col': 'user_id',
    'order_col': 'user_order_seq',
    'target_col': 'outcome',
    'categorical_cols': ['product_type'],
    'drop_cols': ['created_at', 'order_date', 'vendor'],
    'train_ratio': 0.7,
    'val_ratio': 0.2,
    'test_ratio': 0.1,
    'scaler_path': 'models/scaler.pkl',
}

PIPELINE_CONFIG = {'data_path': 'datasets/module_3/feature_frame_filtered.csv'}
