#!/usr/bin/env python3
"""
QKD BB84 Eavesdropping Detection using Machine Learning
========================================================

This script trains Random Forest and SVM classifiers to detect eavesdropping
attacks in BB84 quantum key distribution based on QBER and other metrics.

Author: ML-Enhanced QKD Project
Date: 2025-11-14
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix, 
                             accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, roc_curve)
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


def load_and_prepare_data(filepath):
    """Load QKD dataset and prepare features."""
    print("="*80)
    print("LOADING AND PREPARING DATA")
    print("="*80)

    df = pd.read_excel(filepath)
    print(f"\nDataset loaded: {df.shape[0]} samples, {df.shape[1]} features")
    print("\nColumns in dataset:")
    for col in df.columns:
        print(f"  - {col}")


    # Select important features
    feature_columns = [
        'Raw_sifted_QBER',
        'Shared BER',
        'Full key BER',
        'Per-basis QBER (Z)',
        'Per-basis QBER (X)',
        'Sifted length',
        'Shared key length',
        'Sifting efficiency',
        'Per-basis sifted length (Z)',
        'Per-basis sifted length (X)',
        'Shared differences',
        'Full key differences',
        'Raw_sifted_errors',
        'Raw_sifted_length'
    ]

    print("Columns used for features:")
    for col in feature_columns:
        print(f"  - {col}")

    X = df[feature_columns].copy()
    y = df['Attack label'].copy()

    # Feature Engineering
    print("\nEngineering derived features...")
    X['QBER_mean'] = (X['Per-basis QBER (Z)'] + X['Per-basis QBER (X)']) / 2
    X['QBER_std'] = X[['Per-basis QBER (Z)', 'Per-basis QBER (X)']].std(axis=1)
    X['QBER_ratio'] = X['Per-basis QBER (Z)'] / (X['Per-basis QBER (X)'] + 1e-10)
    X['QBER_asymmetry'] = abs(X['Per-basis QBER (Z)'] - X['Per-basis QBER (X)'])
    X['Error_rate'] = X['Raw_sifted_errors'] / (X['Raw_sifted_length'] + 1e-10)
    X['Key_loss_ratio'] = (X['Raw_sifted_length'] - X['Shared key length']) / (X['Raw_sifted_length'] + 1e-10)
    X['Basis_balance'] = abs(X['Per-basis sifted length (Z)'] - X['Per-basis sifted length (X)']) / \
                         (X['Per-basis sifted length (Z)'] + X['Per-basis sifted length (X)'] + 1e-10)

    # Handle NaN and inf values
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0)

    print(f"Final feature count: {X.shape[1]}")
    print(f"Target distribution: Normal={sum(y==0)}, Attack={sum(y==1)}")

    return X, y, df


def evaluate_baseline(df):
    """Evaluate traditional QKD security assessment."""
    print("\n" + "="*80)
    print("BASELINE: Traditional QKD Security Assessment")
    print("="*80)

    baseline_predictions = []
    for _, row in df.iterrows():
        result = row['Result']
        predicted_attack = 0 if result == "Secure" else 1
        baseline_predictions.append(predicted_attack)

    y_true = df['Attack label'].values
    accuracy = accuracy_score(y_true, baseline_predictions)

    print(f"\nBaseline Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

    cm = confusion_matrix(y_true, baseline_predictions)
    print("\nBaseline Confusion Matrix:")
    print(cm)

    return accuracy, baseline_predictions


def train_random_forest(X, y, n_estimators=100, max_depth=10):
    """Train Random Forest classifier with cross-validation."""
    print("\n" + "="*80)
    print("TRAINING RANDOM FOREST CLASSIFIER")
    print("="*80)

    # Initialize model
    rf_model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=5,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )

    # For larger datasets, use train-test split
    if len(X) >= 100:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        rf_model.fit(X_train, y_train)
        y_pred = rf_model.predict(X_test)
        y_proba = rf_model.predict_proba(X_test)[:, 1]

        print("\nRandom Forest Performance (Test Set):")
        print(f"  Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
        print(f"  Precision: {precision_score(y_test, y_pred, zero_division=0):.4f}")
        print(f"  Recall:    {recall_score(y_test, y_pred, zero_division=0):.4f}")
        print(f"  F1-Score:  {f1_score(y_test, y_pred, zero_division=0):.4f}")

        # Feature importance
        feature_importance = pd.DataFrame({
            'Feature': X.columns,
            'Importance': rf_model.feature_importances_
        }).sort_values('Importance', ascending=False)

        print("\nTop 10 Most Important Features:")
        print(feature_importance.head(10).to_string(index=False))

    else:
        # For small datasets, use cross-validation
        print("\nDataset too small for train-test split. Using cross-validation...")
        rf_model.fit(X, y)

        cv_scores = cross_val_score(rf_model, X, y, cv=min(5, len(X)), 
                                    scoring='accuracy')
        print(f"\nCross-validation Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

        # Feature importance
        feature_importance = pd.DataFrame({
            'Feature': X.columns,
            'Importance': rf_model.feature_importances_
        }).sort_values('Importance', ascending=False)

        print("\nTop 10 Most Important Features:")
        print(feature_importance.head(10).to_string(index=False))

    return rf_model, feature_importance


def train_svm(X, y, C=1.0, kernel='rbf'):
    """Train SVM classifier with cross-validation."""
    print("\n" + "="*80)
    print("TRAINING SUPPORT VECTOR MACHINE (SVM)")
    print("="*80)

    # Standardize features for SVM
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Initialize model
    svm_model = SVC(
        kernel=kernel,
        C=C,
        gamma='scale',
        class_weight='balanced',
        probability=True,
        random_state=42
    )

    # For larger datasets, use train-test split
    if len(X) >= 100:
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )

        svm_model.fit(X_train, y_train)
        y_pred = svm_model.predict(X_test)
        y_proba = svm_model.predict_proba(X_test)[:, 1]

        print("\nSVM Performance (Test Set):")
        print(f"  Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
        print(f"  Precision: {precision_score(y_test, y_pred, zero_division=0):.4f}")
        print(f"  Recall:    {recall_score(y_test, y_pred, zero_division=0):.4f}")
        print(f"  F1-Score:  {f1_score(y_test, y_pred, zero_division=0):.4f}")

    else:
        # For small datasets, use cross-validation
        print("\nDataset too small for train-test split. Using cross-validation...")
        svm_model.fit(X_scaled, y)

        cv_scores = cross_val_score(svm_model, X_scaled, y, cv=min(5, len(X)), 
                                    scoring='accuracy')
        print(f"\nCross-validation Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    return svm_model, scaler


def compare_models(baseline_acc, rf_model, svm_model, X, y):
    """Compare all models and generate summary."""
    print("\n" + "="*80)
    print("MODEL COMPARISON SUMMARY")
    print("="*80)

    comparison = {
        'Model': ['Baseline (Traditional QKD)', 'Random Forest', 'SVM'],
        'Accuracy': [baseline_acc, 'See above', 'See above'],
        'Notes': [
            'Uses Result column',
            'QBER-based ML classification',
            'QBER-based ML classification'
        ]
    }

    comparison_df = pd.DataFrame(comparison)
    print("\n")
    print(comparison_df.to_string(index=False))

    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)
    print("\n1. Collect more data (1000-5000 samples) for robust ML training")
    print("2. Include varying interception densities (0.0 to 1.0)")
    print("3. Balance classes (50% normal, 50% attack)")
    print("4. Test on different noise levels")
    print("5. Consider ensemble methods (combining RF + SVM)")


def main():
    """Main execution function."""
    # Load data
    filepath = 'final_data.xlsx'  # Change to your file path
    X, y, df = load_and_prepare_data(filepath)

    # Evaluate baseline
    baseline_acc, baseline_preds = evaluate_baseline(df)

    # Train Random Forest
    rf_model, rf_importance = train_random_forest(X, y)

    # Train SVM
    svm_model, svm_scaler = train_svm(X, y)

    # Compare models
    compare_models(baseline_acc, rf_model, svm_model, X, y)

    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)

    return rf_model, svm_model, svm_scaler, rf_importance


if __name__ == "__main__":
    rf_model, svm_model, scaler, feature_importance = main()
