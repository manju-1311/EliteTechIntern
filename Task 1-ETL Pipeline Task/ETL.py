import pandas as pd
import numpy as np
import os
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder

def run_advanced_pipeline():
    print("--- 🛠️ ETL PIPELINE STARTED ---")
    
    # 1. EXTRACT: Load the data
    if not os.path.exists('data.csv'):
        print("❌ Error: 'data.csv' not found. Please place it in this folder.")
        return
    
    df = pd.read_csv('data.csv')
    print(f"✅ Step 1: Loaded {df.shape[0]} rows.")

    # 2. TRANSFORM - CLEANING: Remove duplicates and bad rows
    df = df.drop_duplicates()
    df = df.dropna(thresh=3) # Delete rows that are mostly empty
    print("✅ Step 2: Cleaned duplicates and empty rows.")

    # 3. TRANSFORM - FEATURE ENGINEERING (New Feature!)
    # We create a 'Priority_Score' by multiplying Income and Spending (if they exist)
    if 'Annual_Income' in df.columns and 'Spending_Score' in df.columns:
        df['Priority_Score'] = (df['Annual_Income'] / 1000) * df['Spending_Score']
        print("✅ Step 3: Created new feature 'Priority_Score'.")

    # 4. TRANSFORM - HANDLING NUMBERS
    # We fill missing numbers with the MEDIAN (middle value) - it's safer than average
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns
    imputer_num = SimpleImputer(strategy='median')
    df[num_cols] = imputer_num.fit_transform(df[num_cols])

    # Scaling: We squash numbers between 0 and 1 so they look neat on a UI graph
    scaler = MinMaxScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])
    print("✅ Step 4: Numbers filled and scaled (0 to 1).")

    # 5. TRANSFORM - HANDLING TEXT (Encoding)
    # Convert text like "City" into numbers so a computer can read it
    text_cols = df.select_dtypes(include=['object']).columns
    for col in text_cols:
        df[col] = df[col].fillna('Unknown') # Fill missing text
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))
    print("✅ Step 5: Text categories converted to numbers.")

    # 6. LOAD: Save the final result
    df.to_csv('final_report.csv', index=False)
    
    print("--- ✨ PIPELINE COMPLETE ---")
    print(f"📁 Final file saved as: final_report.csv")
    print("\nTop 3 rows of processed data:")
    print(df.head(3))

if __name__ == "__main__":
    run_advanced_pipeline()