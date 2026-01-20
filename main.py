"""
Main orchestration script for the AI/ML pipeline.
Coordinates data loading, preprocessing, model training, and evaluation.

This is the AI implementation.
"""

import os
import sys
import pandas as pd
from pathlib import Path

# Import our custom modules
from dataset_builder import (
    build_training_dataset,
    preprocess_dataset,
    split_dataset
)
from model_training import train_models
from predict_winner import GamePredictor


def main():
    """
    Main pipeline:
    1. Load and aggregate game data from Training Data folder
    2. Preprocess the dataset
    3. Train Random Forest model
    4. Evaluate model performance
    5. Save the model
    """
    
    print("\n" + "="*70)
    print(" NBA GAME PREDICTION - AI/ML PIPELINE")
    print("="*70)
    
    # ===== STEP 1: DEFINE THE PROBLEM =====
    print("\n" + "-"*70)
    print("STEP 1: AI PROBLEM DEFINITION")
    print("-"*70)
    print("""
OBJECTIVE:
    Predict which team will win an NBA game using advanced statistics.

INPUT:
    - Team-level aggregated advanced statistics per game
    - Metrics: TS%, eFG%, ORtg, DRtg, BPM, and more

OUTPUT:
    - Binary prediction: Win (1) or Loss (0) for each team
    """)
    
    # ===== STEP 2: BUILD TRAINING DATASET =====
    print("\n" + "-"*70)
    print("STEP 2: BUILDING TRAINING DATASET")
    print("-"*70)
    
    # Find Training Data directory
    current_dir = Path(__file__).parent
    training_dir = current_dir / "Training Data"
    
    if not training_dir.exists():
        print(f"ERROR: Training Data directory not found at {training_dir}")
        print("Please ensure the Training Data folder contains Excel files with game statistics.")
        return
    
    print(f"Loading games from: {training_dir}")
    
    # Build dataset using reader output
    try:
        df = build_training_dataset(str(training_dir))
        print(f"Loaded {len(df)} team entries from {len(set(df.get('GameID', [])))} games")
    except Exception as e:
        print(f"ERROR: Failed to build dataset: {e}")
        print("Please ensure Excel files have been converted to .xlsx and contain proper data.")
        return
    
    # Check for Result column
    if "Result" not in df.columns or df["Result"].isna().all():
        print("\nWARNING: 'Result' column is missing or all NaN!")
        print("Please ensure game outcomes (Win/Loss) have been populated in the dataset.")
        print("You may need to:")
        print("  1. Add game scores to the Excel files")
        print("  2. Populate the Result column (1=Win, 0=Loss) in the Excel files")
        print("  3. Update dataset_builder.py to extract results from metadata")
        return
    
    print(f"\nDataset shape before preprocessing: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # ===== STEP 3: DATA PREPROCESSING =====
    print("\n" + "-"*70)
    print("STEP 3: DATA PREPROCESSING")
    print("-"*70)
    
    print("3.1 Handling missing values...")
    print("    - Removing players with zero minutes (DNP)")
    print("    - Replacing NaN with column means")
    
    print("3.2 Normalizing features...")
    print("    - Applying MinMaxScaler to numeric features")
    
    df_processed, scaler = preprocess_dataset(df)
    print(f"Dataset shape after preprocessing: {df_processed.shape}")
    
    # Check for missing values
    missing_count = df_processed.isna().sum().sum()
    if missing_count > 0:
        print(f"WARNING: {missing_count} missing values remaining after preprocessing")
    else:
        print("No missing values")
    
    # ===== STEP 4: TRAIN/TEST SPLIT =====
    print("\n" + "-"*70)
    print("STEP 4: TRAIN/TEST SPLIT")
    print("-"*70)
    
    X_train, X_test, y_train, y_test = split_dataset(df_processed, test_size=0.25)
    
    print(f"Training set size: {len(X_train)} samples (75%)")
    print(f"Testing set size: {len(X_test)} samples (25%)")
    print(f"Number of features: {X_train.shape[1]}")
    print(f"Class distribution (Train): {y_train.value_counts().to_dict()}")
    print(f"Class distribution (Test): {y_test.value_counts().to_dict()}")
    
    # ===== STEP 5: TRAIN RANDOM FOREST MODEL =====
    print("\n" + "-"*70)
    print("STEP 5: TRAINING RANDOM FOREST MODEL")
    print("-"*70)
    
    results = train_models(df_processed)
    
    # ===== STEP 6: MODEL EVALUATION SUMMARY =====
    print("\n" + "="*70)
    print("FINAL RESULTS & EVALUATION SUMMARY")
    print("="*70)
    
    rf_acc = results['rf_accuracy']
    
    print(f"\n5.1 MODEL ACCURACY:")
    print(f"    Random Forest: {rf_acc:.4f} ({rf_acc*100:.2f}%)")
    
    print(f"\n5.2 CONFUSION MATRIX:")
    print(f"    {results['rf_confusion_matrix']}")
    
    print(f"\n5.3 FEATURE IMPORTANCE:")
    print(f"    Random Forest - Top 5 Features:")
    for idx, row in results['rf_feature_importance'].head(5).iterrows():
        print(f"      {row['Feature']}: {row['Importance']:.4f}")
    
    print("\n" + "-"*70)
    print("RANDOM FOREST MODEL DETAILS")
    print("-"*70)
    print("See MODEL_DOCUMENTATION.txt for detailed explanation")
    
    print("\n" + "="*70)
    print("PIPELINE COMPLETE - Model ready for UI")
    print("="*70)
    
    # Verify model file exists
    if os.path.exists("trained_model.pkl"):
        print("trained_model.pkl saved and ready for deployment")
    
    return results


if __name__ == "__main__":
    results = main()
