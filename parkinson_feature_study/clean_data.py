#!/usr/bin/env python
"""
Data cleaning script to fix malformed values in the CSV file
"""

import pandas as pd
import numpy as np
import re

def clean_csv_data(input_file, output_file):
    """Clean malformed values in CSV file"""
    
    # Read the file as text first to fix malformed values
    with open(input_file, 'r') as f:
        content = f.read()
    
    # Fix malformed scientific notation (e.g., "1.13#4" -> "1.13E-4")
    # Pattern: number followed by # followed by number
    content = re.sub(r'(\d+\.?\d*)#(\d+)', r'\1E-\2', content)
    
    # Fix any remaining # characters by replacing with E-
    content = content.replace('#', 'E-')
    
    # Write cleaned content to temporary file
    temp_file = input_file.replace('.csv', '_temp.csv')
    with open(temp_file, 'w') as f:
        f.write(content)
    
    # Now read with pandas and handle any remaining issues
    try:
        df = pd.read_csv(temp_file)
        
        # Convert all feature columns to numeric, replacing errors with NaN
        feature_cols = [col for col in df.columns if col not in ['id', 'gender', 'class']]
        
        for col in feature_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Fill NaN values with column median
        for col in feature_cols:
            if df[col].isna().any():
                median_val = df[col].median()
                df[col].fillna(median_val, inplace=True)
                print(f"Filled {df[col].isna().sum()} NaN values in {col} with median {median_val}")
        
        # Save cleaned data
        df.to_csv(output_file, index=False)
        print(f"Cleaned data saved to {output_file}")
        
        # Clean up temp file
        import os
        os.remove(temp_file)
        
        return df
        
    except Exception as e:
        print(f"Error processing file: {e}")
        return None

if __name__ == "__main__":
    input_file = "data/raw/pd_speech_features.csv"
    output_file = "data/raw/pd_speech_features_cleaned.csv"
    
    print("Cleaning dataset...")
    df = clean_csv_data(input_file, output_file)
    
    if df is not None:
        print(f"Dataset shape: {df.shape}")
        print(f"Columns: {len(df.columns)}")
        print("Data types:")
        print(df.dtypes.value_counts())