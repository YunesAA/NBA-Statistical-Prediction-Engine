import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
import numpy as np

def train_models(df):
    """
    Train Random Forest model on the dataset.
    
    Args:
        df: Training dataset with 'Result', 'Team' columns
    
    Returns:
        Dictionary containing model, metrics, and evaluation data
    """
    
    X = df.drop(columns=["Result", "Team", "GameID"], errors="ignore")
    y = df["Result"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # ===== RANDOM FOREST CLASSIFIER (OPTIMIZED) =====
    print("\n" + "="*60)
    print("RANDOM FOREST CLASSIFIER (OPTIMIZED)")
    print("="*60)
    
    # Optimized parameters from hyperparameter tuning
    rf = RandomForestClassifier(
        n_estimators=52,
        max_depth=5,
        min_samples_split=3,
        min_samples_leaf=4,
        random_state=4940,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_acc = accuracy_score(y_test, rf_pred)
    
    print(f"\nAccuracy: {rf_acc:.4f} ({rf_acc*100:.2f}%)")
    print("\nConfusion Matrix:")
    rf_cm = confusion_matrix(y_test, rf_pred)
    print(rf_cm)
    
    print("\nClassification Report:")
    print(classification_report(y_test, rf_pred, target_names=["Loss", "Win"]))
    
    # Feature importance for Random Forest
    rf_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': rf.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print("\nTop 10 Important Features:")
    print(rf_importance.head(10))

    # Save the model
    joblib.dump(rf, "trained_model.pkl")
    print(f"\nSaved model → trained_model.pkl")

    # Return results dictionary
    results = {
        'model': rf,
        'model_name': "Random Forest",
        'accuracy': rf_acc,
        'confusion_matrix': rf_cm,
        'feature_importance': rf_importance,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        # Keep legacy keys for compatibility
        'rf_model': rf,
        'rf_accuracy': rf_acc,
        'rf_confusion_matrix': rf_cm,
        'rf_feature_importance': rf_importance,
        'best_model': rf,
        'best_model_name': "Random Forest",
        'dt_accuracy': 0,  # Placeholder for compatibility
        'dt_confusion_matrix': None,  # Placeholder for compatibility
        'dt_feature_importance': None  # Placeholder for compatibility
    }
    
    return results
