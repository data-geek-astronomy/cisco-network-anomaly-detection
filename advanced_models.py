"""
Advanced Anomaly Detection Models
Using ensemble and deep learning-inspired approaches with scikit-learn
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import (
    IsolationForest, RandomForestClassifier, GradientBoostingClassifier
)
from sklearn.svm import OneClassSVM
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import pickle
import os
from baseline_models import FeatureEngineer, evaluate_model


class SequenceFeatureExtractor:
    """Extract sequence-based features for improved anomaly detection."""

    def __init__(self, window_size=10):
        self.window_size = window_size

    def extract_sequence_features(self, df: pd.DataFrame) -> np.ndarray:
        """Extract temporal and sequence features."""
        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)

        features = []
        for i in range(len(df)):
            start_idx = max(0, i - self.window_size)
            window = df.iloc[start_idx:i+1]

            if len(window) == 0:
                continue

            feature_dict = {
                # Time-based features
                'hour': window['timestamp'].dt.hour.iloc[-1],
                'severity_score': {'INFO': 1, 'WARNING': 2, 'ERROR': 3, 'CRITICAL': 4}[window['severity'].iloc[-1]],
                # Window statistics
                'window_severity_mean': window['severity'].map({'INFO': 1, 'WARNING': 2, 'ERROR': 3, 'CRITICAL': 4}).mean(),
                'window_severity_max': window['severity'].map({'INFO': 1, 'WARNING': 2, 'ERROR': 3, 'CRITICAL': 4}).max(),
                'window_error_count': (window['severity'] != 'INFO').sum(),
                'window_anomaly_ratio': window['is_anomaly'].mean(),
                'window_device_diversity': window['device'].nunique(),
                'window_message_diversity': window['message'].nunique(),
                # Sequence patterns
                'recent_anomalies': window[window['is_anomaly'] == 1].shape[0],
                'consecutive_errors': (window['severity'].isin(['ERROR', 'CRITICAL'])).sum(),
            }

            features.append(feature_dict)

        return pd.DataFrame(features)


class AdvancedAnomalyDetector:
    """Advanced ensemble detector combining multiple algorithms."""

    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train multiple models."""
        X_scaled = self.scaler.fit_transform(X)

        # Random Forest
        self.models['rf'] = RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            random_state=42,
            n_jobs=-1
        ).fit(X_scaled, y)

        # Gradient Boosting
        self.models['gb'] = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        ).fit(X_scaled, y)

        # Isolation Forest (unsupervised baseline)
        self.models['if'] = IsolationForest(
            contamination=0.08,
            random_state=42
        ).fit(X_scaled)

        # One-Class SVM
        self.models['oc_svm'] = OneClassSVM(
            kernel='rbf',
            gamma='auto',
            nu=0.08
        ).fit(X_scaled)

        return self

    def predict(self, X: np.ndarray, method='ensemble') -> np.ndarray:
        """Make predictions using specified method."""
        X_scaled = self.scaler.transform(X)

        if method == 'random_forest':
            return self.models['rf'].predict(X_scaled)
        elif method == 'gradient_boosting':
            return self.models['gb'].predict(X_scaled)
        elif method == 'isolation_forest':
            pred = self.models['if'].predict(X_scaled)
            return (pred == -1).astype(int)
        elif method == 'ocsvm':
            pred = self.models['oc_svm'].predict(X_scaled)
            return (pred == -1).astype(int)
        elif method == 'ensemble':
            # Soft voting ensemble
            rf_pred = self.models['rf'].predict_proba(X_scaled)[:, 1]
            gb_pred = self.models['gb'].predict_proba(X_scaled)[:, 1]
            if_pred = (self.models['if'].predict(X_scaled) == -1).astype(float)
            svm_pred = (self.models['oc_svm'].predict(X_scaled) == -1).astype(float)

            # Average probabilities
            ensemble_score = (rf_pred + gb_pred + if_pred + svm_pred) / 4
            return (ensemble_score > 0.5).astype(int)
        else:
            raise ValueError(f"Unknown method: {method}")

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get probability scores for ensemble."""
        X_scaled = self.scaler.transform(X)

        rf_pred = self.models['rf'].predict_proba(X_scaled)[:, 1]
        gb_pred = self.models['gb'].predict_proba(X_scaled)[:, 1]
        if_pred = (self.models['if'].predict(X_scaled) == -1).astype(float)
        svm_pred = (self.models['oc_svm'].predict(X_scaled) == -1).astype(float)

        return (rf_pred + gb_pred + if_pred + svm_pred) / 4


def main():
    """Train and evaluate advanced models."""
    print("Loading training data...")
    train_df = pd.read_csv('data/train_logs.csv')
    test_df = pd.read_csv('data/test_logs.csv')

    print(f"Training set: {len(train_df)} logs ({train_df['is_anomaly'].sum()} anomalies)")
    print(f"Test set: {len(test_df)} logs ({test_df['is_anomaly'].sum()} anomalies)")

    # Feature engineering
    print("\nFeature engineering...")
    fe = FeatureEngineer()
    train_features = fe.fit_transform(train_df)
    test_features = fe.transform(test_df)

    # Sequence features
    print("Extracting sequence features...")
    seq_extractor = SequenceFeatureExtractor(window_size=10)
    train_seq = seq_extractor.extract_sequence_features(train_features)
    test_seq = seq_extractor.extract_sequence_features(test_features)

    # Combine base features with sequence features
    X_train = train_seq[train_seq.columns].fillna(0).values
    y_train = train_features.iloc[train_seq.index]['is_anomaly'].values

    X_test = test_seq[test_seq.columns].fillna(0).values
    y_test = test_features.iloc[test_seq.index]['is_anomaly'].values

    print(f"Combined feature shape: {X_train.shape}")
    print(f"Training samples with positive class: {y_train.sum()}")

    # Train advanced models
    print("\nTraining advanced ensemble models...")
    detector = AdvancedAnomalyDetector()
    detector.fit(X_train, y_train)

    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)

    # Evaluate models
    print("\nEvaluating advanced models on test set...")
    results = {}

    methods = ['random_forest', 'gradient_boosting', 'isolation_forest', 'ocsvm', 'ensemble']

    for method in methods:
        y_pred = detector.predict(X_test, method=method)
        metrics = evaluate_model(y_test, y_pred, f"Advanced Model - {method.replace('_', ' ').title()}")
        results[method] = metrics

    # Save best model
    with open('models/advanced_detector.pkl', 'wb') as f:
        pickle.dump((detector, fe), f)
    print("\n✓ Models saved to models/advanced_detector.pkl")

    # Save results
    results_df = pd.DataFrame(results).T
    results_df.to_csv('results/advanced_results.csv')
    print(f"✓ Results saved to results/advanced_results.csv")

    # Summary
    print("\n" + "="*60)
    print("Advanced Model Comparison")
    print("="*60)
    summary = results_df[['precision', 'recall', 'f1', 'roc_auc']].sort_values('f1', ascending=False)
    print(summary)

    return detector, results


if __name__ == '__main__':
    main()
