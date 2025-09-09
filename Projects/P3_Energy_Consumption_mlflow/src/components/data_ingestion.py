import os
import pandas as pd
from pathlib import Path

def main():
    """
    Reads the raw data, cleans it, and creates year-based splits for training and validation.
    Training data: 2007-2008 (1-2 years)
    Validation/Drift detection data: 2009-2010
    """
    # Define paths
    project_root = Path(__file__).resolve().parents[2]
    raw_data_path = project_root / "artifacts" / "data_raw" / "household_power_consumption.txt"
    output_dir = project_root / "artifacts" / "data_ingested"
    
    # Create output directories
    output_dir.mkdir(parents=True, exist_ok=True)
    training_dir = output_dir / "training"
    validation_dir = output_dir / "validation"
    training_dir.mkdir(exist_ok=True)
    validation_dir.mkdir(exist_ok=True)

    print(f"Reading raw data from: {raw_data_path}")

    # Read the data
    df = pd.read_csv(
        raw_data_path,
        sep=';',
        low_memory=False,
        na_values=['?']
    )

    print(f"Original dataset shape: {df.shape}")
    print(f"Date range in raw data: {df['Date'].min()} to {df['Date'].max()}")

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

    # 4. Remove any remaining NaN values
    df.dropna(inplace=True)

    print(f"Cleaned dataset shape: {df.shape}")
    print(f"Date range after cleaning: {df.index.min()} to {df.index.max()}")

    # --- Create Year-based Splits ---
    
    # Training data: 2007-2008 (1-2 years for ML training)
    training_years = [2007, 2008]
    training_data = df[df.index.year.isin(training_years)]
    
    print(f"Training data shape: {training_data.shape}")
    print(f"Training data years: {training_data.index.year.unique()}")
    
    # Save training data
    training_path = training_dir / "training_data_2007_2008.csv"
    training_data.to_csv(training_path)
    print(f"Training data saved to: {training_path}")
    
    # Save individual years for training
    for year in training_years:
        year_data = training_data[training_data.index.year == year]
        year_path = training_dir / f"training_data_{year}.csv"
        year_data.to_csv(year_path)
        print(f"Training data for {year} saved to: {year_path}")

    # Validation/Drift detection data: 2009-2010
    validation_years = [2009, 2010]
    validation_data = df[df.index.year.isin(validation_years)]
    
    print(f"Validation data shape: {validation_data.shape}")
    print(f"Validation data years: {validation_data.index.year.unique()}")
    
    # Save validation data
    validation_path = validation_dir / "validation_data_2009_2010.csv"
    validation_data.to_csv(validation_path)
    print(f"Validation data saved to: {validation_path}")
    
    # Save individual years for validation/drift detection
    for year in validation_years:
        year_data = validation_data[validation_data.index.year == year]
        year_path = validation_dir / f"validation_data_{year}.csv"
        year_data.to_csv(year_path)
        print(f"Validation data for {year} saved to: {year_path}")

    # Save complete dataset for reference
    complete_path = output_dir / "complete_dataset_2006_2010.csv"
    df.to_csv(complete_path)
    print(f"Complete dataset saved to: {complete_path}")

    # Print summary statistics
    print("\n=== Data Summary ===")
    print(f"Total records processed: {len(df):,}")
    print(f"Training records (2007-2008): {len(training_data):,}")
    print(f"Validation records (2009-2010): {len(validation_data):,}")
    print(f"Missing data percentage: {df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100:.2f}%")
    
    print("\n=== Year-wise Record Counts ===")
    for year in sorted(df.index.year.unique()):
        year_count = len(df[df.index.year == year])
        purpose = "Training" if year in training_years else "Validation/Drift Detection"
        print(f"{year}: {year_count:,} records ({purpose})")

    print("\nData ingestion and year-based splitting complete!")


if __name__ == '__main__':
    main()