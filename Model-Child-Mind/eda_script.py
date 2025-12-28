import pandas as pd
import os
import glob
import numpy as np

DATA_DIR = r'd:\Hackathons  & Competitions\Synaptix\Model-Child-Mind\data'

def inspect_data():
    # 1. Tabular Train
    train_df = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
    print("Train Shape:", train_df.shape)
    print("Train Columns:", train_df.columns.tolist())
    print(train_df.head())
    
    # 2. Check Parquet
    parquet_dir = os.path.join(DATA_DIR, 'series_train.parquet')
    # Use glob to find a file inside the subdirectories
    # Structure seems to be series_train.parquet/id=XXXX/part-0.parquet?
    # Let's list one subdir
    subdirs = glob.glob(os.path.join(parquet_dir, '*'))
    if len(subdirs) > 0:
        print(f"Found {len(subdirs)} series directories.")
        sample_dir = subdirs[0]
        # Find parquet file inside
        pq_files = glob.glob(os.path.join(sample_dir, '*.parquet'))
        
        if len(pq_files) > 0:
            pq_path = pq_files[0]
            print(f"Reading sample parquet: {pq_path}")
            ts_df = pd.read_parquet(pq_path)
            print("Time Series Shape:", ts_df.shape)
            print("Time Series Columns:", ts_df.columns.tolist())
            print(ts_df.head())
        else:
            print("No parquet files found in subdir.")
    else:
        print("No series subdirectories found.")

if __name__ == "__main__":
    inspect_data()
