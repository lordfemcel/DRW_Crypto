import pandas as pd

def load_train_test(train_path, test_path):
    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)
    return train_df, test_df

def check_data(df):
    print("Shape:", df.shape)
    print("Missing values per column:\n", df.isnull().sum())
