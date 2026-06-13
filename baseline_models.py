"""
Statistical Baseline Anomaly Detection Models
Implements Z-score, IQR, Isolation Forest, and Local Outlier Factor
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import (
    confusion_matrix, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve, auc
)
import pickle
import os
from datetime import datetime
from typing import Dict, Tuple, Any


class FeatureEngineer:
    """Extract features from raw logs for anomaly detection."""

    def __init__(self):
        self.severity_encoder = LabelEncoder()
        self.device_encoder = LabelEncoder()
        self.message_encoder = LabelEncoder()
        self.fitted = False
        self.feature_cols = [
            'hour', 'day_of_week', 'minute', 'severity_encoded',
            'device_encoded', 'severity_score', 'message_length',
            'device_error_count_rolling', 'hourly_error_count_rolling'
        ]

    def fit(self, df: pd.DataFrame) -> 'FeatureEngineer':
        """Fit encoders on training data."""
        self.severity_encoder.fit(df['severity'])
        self.device_encoder.fit(df['device'])
        self.message_encoder.fit(df['message'])
        self.fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract features from logs."""
        if not self.fitted:
            raise ValueError("FeatureEngineer must be fit before transform")

        df = df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Time-based features
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['minute'] = df['timestamp'].dt.minute

        # Encode categorical variables
        df['severity_encoded'] = self.severity_encoder.transform(df['severity'])
        df['device_encoded'] = self.device_encoder.transform(df['device'])
        df['message_encoded'] = self.message_encoder.transform(df['message'])

        # Severity score (higher = more severe)
        severity_score = {'INFO': 1, 'WARNING': 2, 'ERROR': 3, 'CRITICAL': 4}
        df['severity_score'] = df['severity'].map(severity_score)

        # Message complexity (length-based proxy for complexity)
        df['message_length'] = df['message'].str.len()

        # Device-level features (within rolling window)
        df['device_error_count_rolling'] = (
            df.groupby('device')['severity_score']
            .rolling(window=100, min_periods=1)
            .mean()
            .reset_index(0, drop=True)
        )

        # Time-based error frequency
        df['hourly_error_count_rolling'] = (
            df.groupby('hour')['severity_score']
            .rolling(window=100, min_periods=1)
            .mean()
            .reset_index(0, drop=True)
        )

        return df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(df).transform(df)


class StatisticalAnomalyDetector:
    """Statistical baseline models for anomaly detection."""

    def __init__(self):
        self.scaler = StandardScaler()
        self.models = {}
        self.feature_cols = [
            'hour', 'day_of_week', 'minute', 'severity_encoded',
            'device_encoded', 'severity_score', 'message_length',
            'device_error_count_rolling', 'hourly_error_count_rolling'
        ]

    def fit(self, X: np.ndarray) -> 'StatisticalAnomalyDetector':
        """Fit all models."""
        X_scaled = self.scaler.fit_transform(X)

        # Z-score threshold (dynamic)
        self.z_score_threshold = 3.0

        # Isolation Forest
        self.models['isolation_forest'] = IsolationForest(
            contamination=0.08,
            random_state=42,
            n_jobs=-1
        ).fit(X_scaled)

        # Local Outlier Factor
        self.models['lof'] = LocalOutlierFactor(
            n_neighbors=20,
            contamination=0.08,
            novelty=True,
            n_jobs=-1
        ).fit(X_scaled)

        return self

    def predict_zscore(self, X: np.ndarray) -> np.ndarray:
        """Z-score based anomaly detection."""
        X_scaled = self.scaler.transform(X)
        z_scores = np.abs(X_scaled)
        return (z_scores.max(axis=1) > self.z_score_threshold).astype(int)

    def predict_iqr(self, X: np.ndarray) -> np.ndarray:
        """IQR-based anomaly detection (univariate on severity score)."""
        # Assuming last column is severity_score
        severity = X[:, -3]  # severity_score is -3 from end
        Q1 = np.percentile(severity, 25)
        Q3 = np.percentile(severity, 75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        return ((severity < lower) | (severity > upper)).astype(int)

    def predict_isolation_forest(self, X: np.ndarray) -> np.ndarray:
        """Isolation Forest predictions."""
        X_scaled = self.scaler.transform(X)
        predictions = self.models['isolation_forest'].predict(X_scaled)
        return (predictions == -1).astype(int)

    def predict_lof(self, X: np.ndarray) -> np.ndarray:
        """Local Outlier Factor predictions."""
        X_scaled = self.scaler.transform(X)
        predictions = self.models['lof'].predict(X_scaled)
        return (predictions == -1).astype(int)

    def predict_ensemble(self, X: np.ndarray, voting='majority') -> np.ndarray:
        """Ensemble prediction using multiple methods."""
        zscore_pred = self.predict_zscore(X)
        iqr_pred = self.predict_iqr(X)
        if_pred = self.predict_isolation_forest(X)
        lof_pred = self.predict_lof(X)

        ensemble = np.column_stack([zscore_pred, iqr_pred, if_pred, lof_pred])

        if voting == 'majority':
            return (ensemble.sum(axis=1) > 2).astype(int)
        elif voting == 'all':
            return (ensemble.sum(axis=1) == 4).astype(int)
        else:  # any
            return (ensemble.sum(axis=1) > 0).astype(int)


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray, model_name: str = "") -> Dict[str, float]:
    """Evaluate model performance."""
    metrics = {
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
    }

    # ROC-AUC (if we have probability scores)
    try:
        metrics['roc_auc'] = roc_auc_score(y_true, y_pred)
    except:
        metrics['roc_auc'] = 0.0

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics['true_positives'] = int(tp)
    metrics['false_positives'] = int(fp)
    metrics['false_negatives'] = int(fn)
    metrics['true_negatives'] = int(tn)

    print(f"\n{'='*60}")
    print(f"Model: {model_name}")
    print(f"{'='*60}")
    print(f"Precision: {metrics['precision']:.4f} (TP / (TP + FP))")
    print(f"Recall:    {metrics['recall']:.4f} (TP / (TP + FN))")
    print(f"F1-Score:  {metrics['f1']:.4f}")
    print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"  True Positives:  {metrics['true_positives']}")
    print(f"  False Positives: {metrics['false_positives']}")
    print(f"  False Negatives: {metrics['false_negatives']}")
    print(f"  True Negatives:  {metrics['true_negatives']}")

    return metrics


def main():
    """Train and evaluate baseline models."""
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

    # Extract feature matrix
    X_train = train_features[fe.feature_cols].fillna(0).values
    y_train = train_features['is_anomaly'].values

    X_test = test_features[fe.feature_cols].fillna(0).values
    y_test = test_features['is_anomaly'].values

    print(f"Feature shape: {X_train.shape}")

    # Train baseline models
    print("\nTraining statistical models...")
    detector = StatisticalAnomalyDetector()
    detector.fit(X_train)

    # Evaluate on test set
    print("\nEvaluating models on test set...")

    results = {}

    # Z-Score
    y_pred = detector.predict_zscore(X_test)
    results['z_score'] = evaluate_model(y_test, y_pred, "Z-Score")

    # IQR
    y_pred = detector.predict_iqr(X_test)
    results['iqr'] = evaluate_model(y_test, y_pred, "IQR")

    # Isolation Forest
    y_pred = detector.predict_isolation_forest(X_test)
    results['isolation_forest'] = evaluate_model(y_test, y_pred, "Isolation Forest")

    # Local Outlier Factor
    y_pred = detector.predict_lof(X_test)
    results['lof'] = evaluate_model(y_test, y_pred, "Local Outlier Factor")

    # Ensemble (majority voting)
    y_pred = detector.predict_ensemble(X_test, voting='majority')
    results['ensemble_majority'] = evaluate_model(y_test, y_pred, "Ensemble (Majority Voting)")

    # Ensemble (all voting)
    y_pred = detector.predict_ensemble(X_test, voting='all')
    results['ensemble_all'] = evaluate_model(y_test, y_pred, "Ensemble (All Vote)")

    # Save models
    os.makedirs('models', exist_ok=True)
    with open('models/baseline_detector.pkl', 'wb') as f:
        pickle.dump((detector, fe), f)
    print("\n✓ Models saved to models/baseline_detector.pkl")

    # Save results
    os.makedirs('results', exist_ok=True)
    results_df = pd.DataFrame(results).T
    results_df.to_csv('results/baseline_results.csv')
    print(f"✓ Results saved to results/baseline_results.csv")

    return detector, fe, results


if __name__ == '__main__':
    main()
