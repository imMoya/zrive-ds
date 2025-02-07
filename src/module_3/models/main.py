from dataloader import DataLoader
from pathlib import Path


if __name__ == "__main__":
    # Load the data
    DATA_DIR = Path('datasets/module_3/')
    data_path = Path(DATA_DIR, 'feature_frame_filtered.csv')
    data_loader = DataLoader(data_path)
    df = data_loader.load_data()
    print(df.head())