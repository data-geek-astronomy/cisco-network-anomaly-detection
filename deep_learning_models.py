"""
Deep Learning Models for Anomaly Detection
LSTM Autoencoder and Transformer-based models
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, auc
)
import matplotlib.pyplot as plt
import os
from baseline_models import FeatureEngineer, evaluate_model


class LSTMAutoencoder(keras.Model):
    """LSTM-based autoencoder for sequence anomaly detection."""

    def __init__(self, input_shape, latent_dim=16):
        super(LSTMAutoencoder, self).__init__()
        self.latent_dim = latent_dim
        self.input_shape_model = input_shape

        # Encoder
        self.encoder = keras.Sequential([
            layers.LSTM(64, activation='relu', input_shape=input_shape),
            layers.Dense(latent_dim, activation='relu')
        ])

        # Decoder
        self.decoder = keras.Sequential([
            layers.RepeatVector(input_shape[0]),
            layers.LSTM(64, activation='relu', return_sequences=True),
            layers.TimeDistributed(layers.Dense(input_shape[1]))
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class TransformerAnomalyDetector(keras.Model):
    """Transformer-based model for sequence anomaly detection."""

    def __init__(self, input_shape, num_heads=4, ff_dim=32):
        super(TransformerAnomalyDetector, self).__init__()
        self.input_shape_model = input_shape

        # Input embedding
        self.embedding = layers.Dense(32)

        # Multi-head attention
        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=32 // num_heads
        )

        # Feed forward
        self.ffn = keras.Sequential([
            layers.Dense(ff_dim, activation='relu'),
            layers.Dense(32)
        ])

        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)

        # Output layers
        self.global_pool = layers.GlobalAveragePooling1D()
        self.dense1 = layers.Dense(16, activation='relu')
        self.dense2 = layers.Dense(1, activation='sigmoid')

    def call(self, x):
        # Embedding
        x_embedded = self.embedding(x)

        # Attention with residual connection
        attn_output = self.attention(x_embedded, x_embedded)
        x_attn = self.layernorm1(x_embedded + attn_output)

        # FFN with residual connection
        ffn_output = self.ffn(x_attn)
        x_ffn = self.layernorm2(x_attn + ffn_output)

        # Global pooling and classification
        pooled = self.global_pool(x_ffn)
        dense1_out = self.dense1(pooled)
        output = self.dense2(dense1_out)

        return output


def create_sequences(data, labels, sequence_length=10):
    """Create sliding window sequences from data."""
    X, y = [], []
    for i in range(len(data) - sequence_length + 1):
        X.append(data[i:i + sequence_length])
        # Label is 1 if any log in sequence is anomalous
        y.append(1 if labels[i:i + sequence_length].sum() > 0 else 0)
    return np.array(X), np.array(y)


def train_lstm_autoencoder(X_train, X_test, y_test, sequence_length=10, epochs=20):
    """Train LSTM autoencoder."""
    print("\n" + "="*60)
    print("Training LSTM Autoencoder")
    print("="*60)

    model = LSTMAutoencoder(input_shape=(sequence_length, X_train.shape[2]), latent_dim=16)
    model.compile(optimizer='adam', loss='mse')

    # Train on normal sequences only
    normal_idx = (y_train == 0)
    X_normal = X_train[normal_idx]

    history = model.fit(
        X_normal, X_normal,
        epochs=epochs,
        batch_size=32,
        validation_split=0.1,
        verbose=0
    )

    # Evaluate using reconstruction error
    X_test_recon = model.predict(X_test, verbose=0)
    reconstruction_error = np.mean(np.abs(X_test - X_test_recon), axis=(1, 2))

    # Threshold: 95th percentile of normal error
    threshold = np.percentile(reconstruction_error[y_test == 0], 95)
    y_pred = (reconstruction_error > threshold).astype(int)

    metrics = evaluate_model(y_test, y_pred, "LSTM Autoencoder")
    return model, metrics, reconstruction_error, threshold


def train_transformer(X_train, X_test, y_test, sequence_length=10, epochs=20):
    """Train Transformer-based model."""
    print("\n" + "="*60)
    print("Training Transformer Model")
    print("="*60)

    model = TransformerAnomalyDetector(
        input_shape=(sequence_length, X_train.shape[2]),
        num_heads=4,
        ff_dim=32
    )

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['binary_accuracy']
    )

    # Train on all data
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=32,
        validation_split=0.1,
        verbose=0
    )

    # Get predictions
    y_pred_prob = model.predict(X_test, verbose=0).flatten()
    y_pred = (y_pred_prob > 0.5).astype(int)

    metrics = evaluate_model(y_test, y_pred, "Transformer Model")
    return model, metrics, y_pred_prob


def main():
    """Train and evaluate deep learning models."""
    print("Loading training data...")
    train_df = pd.read_csv('data/train_logs.csv')
    test_df = pd.read_csv('data/test_logs.csv')

    # Feature engineering
    print("Feature engineering...")
    fe = FeatureEngineer()
    train_features = fe.fit_transform(train_df)
    test_features = fe.transform(test_df)

    # Extract features
    X_train_raw = train_features[fe.feature_cols].fillna(0).values
    y_train_raw = train_features['is_anomaly'].values

    X_test_raw = test_features[fe.feature_cols].fillna(0).values
    y_test_raw = test_features['is_anomaly'].values

    # Normalize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)

    # Create sequences
    sequence_length = 10
    print(f"\nCreating sequences (window size: {sequence_length})...")
    X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_raw, sequence_length)
    X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test_raw, sequence_length)

    print(f"Train sequences: {X_train_seq.shape}")
    print(f"Test sequences: {X_test_seq.shape}")
    print(f"Positive samples in train: {y_train_seq.sum()} ({y_train_seq.mean()*100:.1f}%)")
    print(f"Positive samples in test: {y_test_seq.sum()} ({y_test_seq.mean()*100:.1f}%)")

    # Train models
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)

    results = {}

    # LSTM Autoencoder
    lstm_model, lstm_metrics, _, _ = train_lstm_autoencoder(
        X_train_seq, X_test_seq, y_test_seq,
        sequence_length=sequence_length,
        epochs=20
    )
    results['lstm_autoencoder'] = lstm_metrics
    lstm_model.save('models/lstm_autoencoder.h5')
    print("✓ LSTM model saved")

    # Transformer
    transformer_model, transformer_metrics, _ = train_transformer(
        X_train_seq, X_test_seq, y_test_seq,
        sequence_length=sequence_length,
        epochs=20
    )
    results['transformer'] = transformer_metrics
    transformer_model.save('models/transformer_model.h5')
    print("✓ Transformer model saved")

    # Save results
    results_df = pd.DataFrame(results).T
    results_df.to_csv('results/deep_learning_results.csv')
    print("\n✓ Results saved to results/deep_learning_results.csv")

    # Comparison
    print("\n" + "="*60)
    print("Model Comparison Summary")
    print("="*60)
    print(results_df[['precision', 'recall', 'f1']])

    return lstm_model, transformer_model, results


if __name__ == '__main__':
    main()
