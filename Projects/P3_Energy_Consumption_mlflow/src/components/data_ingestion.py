import os
import pandas as pd
from pathlib import Path

def main():
    """
    Reads the raw data, cleans it, and saves a sample for 2007.
    """
    # Define paths
    project_root = Path(__file__).resolve().parents[2]
    raw_data_path = project_root / "artifacts" / "data_raw" / "household_power_consumption.txt"
    output_dir = project_root / "artifacts" / "data_ingested"
    output_path = output_dir / "data_2007.csv"

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading raw data from: {raw_data_path}")

    # Read the data
    df = pd.read_csv(
        raw_data_path,
        sep=';',
        low_memory=False,
        na_values=['?']
    )

    # --- Data Cleaning and Transformation ---
    # 1. Combine Date and Time into a single datetime column
    df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], dayfirst=True)
    df = df.set_index('datetime')
    df = df.drop(['Date', 'Time'], axis=1)

    # 2. Convert all data columns to numeric, coercing errors
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # 3. Handle missing values (using forward fill for time series)
    df.ffill(inplace=True)

    # 4. Filter for the year 2007 to create a smaller sample
    df_2007 = df[df.index.year == 2007]

    print(f"Saving 2007 data sample to: {output_path}")
    df_2007.to_csv(output_path)
    print("Data ingestion and sampling complete.")


if __name__ == '__main__':
    main()