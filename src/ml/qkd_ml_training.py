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

import os
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

    # Convert Num_bits to numeric if not already for latter filtering
    df['Num_bits'] = pd.to_numeric(df['Num_bits'], errors='coerce')


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
    
    feature_columns += [
        'QBER_mean',
        'QBER_std',
        'QBER_ratio',
        'QBER_asymmetry',
        'Error_rate',
        'Key_loss_ratio',
        'Basis_balance'
    ]
    print(f"Total features after engineering: {len(feature_columns)}")

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
    precision = precision_score(y_true, baseline_predictions, zero_division=0)
    recall_v = recall_score(y_true, baseline_predictions, zero_division=0)
    f1 = f1_score(y_true, baseline_predictions, zero_division=0)

    print(f"\nBaseline Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Baseline Precision: {precision:.4f}")
    print(f"Baseline Recall:    {recall_v:.4f}")
    print(f"Baseline F1-Score:  {f1:.4f}")

    cm = confusion_matrix(y_true, baseline_predictions)
    print("\nBaseline Confusion Matrix:")
    print(cm)

    return accuracy, baseline_predictions, recall_v, f1


def train_random_forest(X, y, n_estimators=100, max_depth=10):
    """Train Random Forest classifier with cross-validation.

    Returns
    -------
    rf_model : RandomForestClassifier
        Trained RF model.
    feature_importance : pd.DataFrame
        Feature importances.
    rf_accuracy : float
        Accuracy from test split (if len>=100) or CV mean accuracy otherwise.
    """
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
    rf_accuracy = None
    rf_recall = None

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

        rf_accuracy = accuracy_score(y_test, y_pred)
        rf_recall = recall_score(y_test, y_pred, zero_division=0)

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
        rf_accuracy = float(cv_scores.mean())
        # Approximate recall via CV using scoring='recall' for completeness
        try:
            cv_recall = cross_val_score(rf_model, X, y, cv=min(5, len(X)), scoring='recall')
            rf_recall = float(cv_recall.mean())
            print(f"Cross-validation Recall:   {rf_recall:.4f} (+/- {cv_recall.std():.4f})")
        except Exception:
            rf_recall = None

        # Feature importance
        feature_importance = pd.DataFrame({
            'Feature': X.columns,
            'Importance': rf_model.feature_importances_
        }).sort_values('Importance', ascending=False)

        print("\nTop 10 Most Important Features:")
        print(feature_importance.head(10).to_string(index=False))

    return rf_model, feature_importance, rf_accuracy, rf_recall


def train_svm(X, y, C=1.0, kernel='rbf'):
    """Train SVM classifier with cross-validation.

    Returns
    -------
    svm_model : SVC
        Trained SVM model.
    scaler : StandardScaler
        Fitted scaler for feature standardization.
    svm_accuracy : float
        Accuracy from test split (if len>=100) or CV mean accuracy otherwise.
    """
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
    svm_accuracy = None
    svm_recall = None

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

        svm_accuracy = accuracy_score(y_test, y_pred)
        svm_recall = recall_score(y_test, y_pred, zero_division=0)

    else:
        # For small datasets, use cross-validation
        print("\nDataset too small for train-test split. Using cross-validation...")
        svm_model.fit(X_scaled, y)

        cv_scores = cross_val_score(svm_model, X_scaled, y, cv=min(5, len(X)), 
                                    scoring='accuracy')
        print(f"\nCross-validation Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        svm_accuracy = float(cv_scores.mean())
        try:
            cv_rec = cross_val_score(svm_model, X_scaled, y, cv=min(5, len(X)), scoring='recall')
            svm_recall = float(cv_rec.mean())
            print(f"Cross-validation Recall:   {svm_recall:.4f} (+/- {cv_rec.std():.4f})")
        except Exception:
            svm_recall = None
    return svm_model, scaler, svm_accuracy, svm_recall


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


def do_training(X, y, df):
    # Evaluate baseline
    baseline_acc, baseline_preds, baseline_rec, baseline_f1 = evaluate_baseline(df)

    # Train Random Forest
    rf_model, rf_importance, rf_acc, rf_rec = train_random_forest(X, y)

    # Train SVM
    svm_model, svm_scaler, svm_acc, svm_rec = train_svm(X, y)

    # Compare models (prints)
    compare_models(baseline_acc, rf_model, svm_model, X, y)

    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)

    return {
        'baseline_acc': baseline_acc,
        'baseline_recall': baseline_rec,
        'baseline_f1': baseline_f1,
        'rf_acc': rf_acc,
        'rf_recall': rf_rec,
        'svm_acc': svm_acc,
        'svm_recall': svm_rec,
        'rf_importance': rf_importance
    }

def main():
    """Main execution function."""
    # Load data
    dataFolder = os.path.join(os.path.dirname(__file__), 'data')
    filepath = os.path.join(dataFolder, 'final_data.xlsx')  # Change to your file path
    X, y, df = load_and_prepare_data(filepath)

    ### Do training with full dataset
    print("\n" + "="*80)
    print("TRAINING WITH FULL DATASET")
    print("="*80)
    full_metrics = do_training(X, y, df)


    ### Do training with decreasing Num_bits ranges and compare
    ranges = [
        (20, 180),
        (20, 100),
        (20, 60),
        (20, 50),
        (20, 40),
        (20, 30),
        (20, 25)
    ]

    results = []
    for low, high in ranges:
        print("\n" + "="*80)
        print(f"TRAINING WITH FILTERED DATA (Num_bits between {low} and {high})")
        print("="*80)

        mask = (df['Num_bits'] >= low) & (df['Num_bits'] <= high)
        X_f = X[mask].reset_index(drop=True)
        y_f = y[mask].reset_index(drop=True)
        df_f = df[mask].reset_index(drop=True)

        print(f"\nFiltered dataset: {X_f.shape[0]} samples with Num_bits between {low}-{high}")

        metrics = do_training(X_f, y_f, df_f)
        results.append({
            'Range': f"{low}-{high}",
            'Samples': int(X_f.shape[0]),
            'Baseline_Acc': metrics['baseline_acc'],
            'Baseline_Recall': metrics['baseline_recall'],
            'Baseline_F1': metrics['baseline_f1'],
            'RF_Acc': metrics['rf_acc'],
            'RF_Recall': metrics['rf_recall'],
            'SVM_Acc': metrics['svm_acc'],
            'SVM_Recall': metrics['svm_recall'],
        })

    # Aggregate and print comparison table
    print("\n" + "="*80)
    print("ACCURACY COMPARISON ACROSS NUM_BITS RANGES")
    print("="*80)
    summary_df = pd.DataFrame(results)
    print(summary_df.to_string(index=False))

    # Save CSV to data/ml_outputs to align with notebook
    out_dir = os.path.join(os.path.dirname(__file__), 'data', 'ml_outputs')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'accuracy_comparison_by_num_bits.csv')
    summary_df.to_csv(out_path, index=False)
    print(f"\nSaved comparison CSV: {out_path}")


if __name__ == "__main__":
    main()
