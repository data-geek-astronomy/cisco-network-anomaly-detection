"""
Synthetic Network Infrastructure Log Generator
Generates realistic syslog-format logs with natural patterns and injected anomalies
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
from typing import List, Dict, Tuple

class NetworkLogGenerator:
    """Generate synthetic network infrastructure logs with realistic patterns and anomalies."""

    def __init__(self, seed: int = 42):
        np.random.seed(seed)
        random.seed(seed)

        # Network infrastructure components
        self.device_ips = [
            '192.168.1.1', '192.168.1.2', '192.168.1.5', '192.168.1.10',
            '10.0.0.1', '10.0.0.5', '10.0.0.15', '10.0.0.20',
            '172.16.0.1', '172.16.0.10', '172.16.0.50'
        ]

        self.device_names = [
            'router-core-01', 'router-core-02', 'switch-dist-01', 'switch-dist-02',
            'firewall-01', 'firewall-02', 'gateway-01', 'gateway-02',
            'ap-01', 'ap-02', 'ap-03'
        ]

        self.log_events = {
            'INFO': [
                'Interface up',
                'BGP session established',
                'OSPF neighbor formed',
                'ARP resolved',
                'DNS query successful',
                'Authentication successful',
                'Configuration saved'
            ],
            'WARNING': [
                'High CPU utilization',
                'Memory usage above threshold',
                'Link utilization high',
                'Packet loss detected',
                'Latency increased',
                'Connection reset',
                'Duplicate IP detected'
            ],
            'ERROR': [
                'Interface down',
                'BGP session failed',
                'Connection timeout',
                'Authentication failed',
                'Device unreachable',
                'Packet dropped',
                'Memory error'
            ],
            'CRITICAL': [
                'Device offline',
                'Complete link failure',
                'Multiple interface failures',
                'System overload',
                'Critical service down',
                'Hardware failure detected'
            ]
        }

        self.normal_severity_dist = {'INFO': 0.6, 'WARNING': 0.3, 'ERROR': 0.08, 'CRITICAL': 0.02}
        self.anomaly_severity_dist = {'INFO': 0.1, 'WARNING': 0.2, 'ERROR': 0.4, 'CRITICAL': 0.3}

    def _generate_timestamp_sequence(self, num_logs: int, start_date: datetime) -> List[datetime]:
        """Generate timestamps with realistic log arrival patterns."""
        timestamps = []
        current_time = start_date

        for _ in range(num_logs):
            # Add time between logs (1s to 30s, realistic for network traffic)
            time_delta = np.random.exponential(scale=5)  # Average 5 seconds between logs
            current_time += timedelta(seconds=time_delta)
            timestamps.append(current_time)

        return timestamps

    def _get_severity_dist(self, is_anomaly: bool) -> str:
        """Sample severity based on whether this is anomaly or normal."""
        dist = self.anomaly_severity_dist if is_anomaly else self.normal_severity_dist
        return np.random.choice(
            list(dist.keys()),
            p=list(dist.values())
        )

    def _get_message(self, severity: str, is_anomaly: bool) -> str:
        """Get appropriate log message."""
        if is_anomaly:
            # For anomalies, bias toward WARNING/ERROR/CRITICAL
            return random.choice(
                self.log_events['ERROR'] + self.log_events['WARNING'] + self.log_events['CRITICAL']
            )
        else:
            return random.choice(self.log_events[severity])

    def _get_source_device(self, device_sequence: np.ndarray, current_idx: int, is_anomaly: bool) -> str:
        """Get source device with temporal correlation."""
        if is_anomaly:
            # Anomalies tend to cluster on specific devices (realistic)
            if np.random.random() < 0.7 and current_idx > 0:
                return self.device_names[device_sequence[current_idx - 1]]
            else:
                return random.choice(self.device_names)
        else:
            # Normal logs spread across devices
            if np.random.random() < 0.3:
                return self.device_names[device_sequence[current_idx - 1]]
            else:
                return random.choice(self.device_names)

    def generate_logs(
        self,
        num_logs: int = 100000,
        anomaly_ratio: float = 0.08,
        start_date: datetime = None
    ) -> pd.DataFrame:
        """
        Generate synthetic network logs.

        Args:
            num_logs: Number of logs to generate
            anomaly_ratio: Percentage of logs that are anomalies (0-1)
            start_date: Starting timestamp (default: 30 days ago)

        Returns:
            DataFrame with columns: timestamp, severity, device, source_ip, message, is_anomaly
        """
        if start_date is None:
            start_date = datetime.now() - timedelta(days=30)

        # Determine which logs are anomalies
        num_anomalies = int(num_logs * anomaly_ratio)
        is_anomaly_array = np.array([False] * (num_logs - num_anomalies) + [True] * num_anomalies)
        np.random.shuffle(is_anomaly_array)

        # Pre-allocate device sequence for temporal correlation
        device_sequence = np.random.randint(0, len(self.device_names), size=num_logs)

        # Generate timestamps
        timestamps = self._generate_timestamp_sequence(num_logs, start_date)

        # Generate logs
        logs = []
        for i in range(num_logs):
            is_anomaly = is_anomaly_array[i]
            severity = self._get_severity_dist(is_anomaly)
            device = self._get_source_device(device_sequence, i, is_anomaly)
            source_ip = self.device_ips[device_sequence[i]]
            message = self._get_message(severity, is_anomaly)

            logs.append({
                'timestamp': timestamps[i],
                'severity': severity,
                'device': device,
                'source_ip': source_ip,
                'message': message,
                'is_anomaly': int(is_anomaly)
            })

        df = pd.DataFrame(logs)
        return df.sort_values('timestamp').reset_index(drop=True)

    def save_logs(self, df: pd.DataFrame, filepath: str) -> None:
        """Save logs to CSV."""
        df.to_csv(filepath, index=False)
        print(f"✓ Saved {len(df)} logs to {filepath}")
        print(f"  Anomalies: {df['is_anomaly'].sum()} ({df['is_anomaly'].mean()*100:.1f}%)")
        print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")


def main():
    """Generate synthetic logs and save to CSV."""
    import os

    generator = NetworkLogGenerator(seed=42)

    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)

    # Generate training set (80k logs)
    print("Generating training logs...")
    train_df = generator.generate_logs(
        num_logs=80000,
        anomaly_ratio=0.08,
        start_date=datetime.now() - timedelta(days=25)
    )
    generator.save_logs(train_df, 'data/train_logs.csv')

    # Generate test set (20k logs)
    print("\nGenerating test logs...")
    test_df = generator.generate_logs(
        num_logs=20000,
        anomaly_ratio=0.10,
        start_date=datetime.now() - timedelta(days=5)
    )
    generator.save_logs(test_df, 'data/test_logs.csv')

    print("\n✓ Dataset generation complete!")
    print(f"Training set shape: {train_df.shape}")
    print(f"Test set shape: {test_df.shape}")


if __name__ == '__main__':
    main()
