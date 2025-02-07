from dataloader import DataLoader
from preprocessor import Preprocessor
from pathlib import Path


if __name__ == '__main__':
    # Load the data
    DATA_DIR = Path('datasets/module_3/')
    data_path = Path(DATA_DIR, 'feature_frame_filtered.csv')
    data_loader = DataLoader(data_path)
    df = data_loader.load_data()
    # Preprocess the data
    preprocessor = Preprocessor()
    (X_train_scaled, y_train), (X_val_scaled, y_val), (X_test_scaled, y_test) = (
        preprocessor.fit_transform(df)
    )