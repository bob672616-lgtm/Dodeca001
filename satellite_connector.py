"""
satellite_connector.py

Provides a SatelliteConnector class to fetch or simulate satellite telemetry
and imagery-derived feature vectors for training integration.

Modes:
- simulate: generate synthetic, physically-inspired feature vectors
- api: (placeholder) show how to call real APIs (NASA, Sentinel, etc.)

Usage:
    from satellite_connector import SatelliteConnector
    connector = SatelliteConnector(simulate=True)
    X_sat, Y_sat = connector.fetch_latest_features(num_samples=200)

X_sat: torch.Tensor shape (num_samples, feature_dim)
Y_sat: torch.Tensor shape (num_samples, target_dim)
"""

from typing import Tuple, Optional
import os
import numpy as np
import torch
import requests

class SatelliteConnector:
    def __init__(self, simulate: bool = True, source: str = "simulated", api_key: Optional[str] = None):
        self.simulate = simulate
        self.source = source
        self.api_key = api_key

    def fetch_latest_features(self, num_samples: int = 200, feature_dim: int = 384, target_dim: int = 32) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return a pair (X, Y) of features and targets suitable for appending to training data.

        When `simulate` is True, generates noisy but structured vectors that mimic
        spectral / embedding patterns from satellite-derived observations.
        """
        if self.simulate:
            return self._simulate_features(num_samples, feature_dim, target_dim)
        else:
            # Try a simple live call to NASA Earth imagery API as an example.
            # If it fails for any reason, fall back to simulated data.
            try:
                if not self.api_key:
                    raise ValueError("API key required for live mode")

                # Use a couple of sample coordinates (lat, lon) - this is illustrative.
                samples = []
                for i in range(min(num_samples, 5)):
                    lat = 0.0 + i * 1.0
                    lon = 0.0 + i * 1.0
                    url = 'https://api.nasa.gov/planetary/earth/imagery'
                    params = {'lat': lat, 'lon': lon, 'date': '2018-01-01', 'api_key': self.api_key}
                    resp = requests.get(url, params=params, timeout=5)
                    if resp.status_code == 200:
                        # We won't process imagery bytes here; instead record that call succeeded
                        samples.append((lat, lon, True))
                    else:
                        samples.append((lat, lon, False))

                # If we got at least one successful call, synthesize features biased by success
                success_count = sum(1 for s in samples if s[2])
                if success_count > 0:
                    # bias simulation by success_count
                    X, Y = self._simulate_features(num_samples, feature_dim, target_dim)
                    X += torch.randn_like(X) * (0.1 * (1.0 - success_count / len(samples)))
                    return X, Y
                else:
                    # fallback
                    return self._simulate_features(num_samples, feature_dim, target_dim)
            except Exception:
                return self._simulate_features(num_samples, feature_dim, target_dim)

    def _simulate_features(self, num_samples: int, feature_dim: int, target_dim: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Create base patterns that vary smoothly (simulate seasonal / spatial signatures)
        t = np.linspace(0, 2 * np.pi, feature_dim)
        base_patterns = np.stack([np.sin(t * (1 + i * 0.03)) for i in range(num_samples)], axis=0)

        # Add per-sample variability and realistic noise
        noise = np.random.normal(scale=0.2, size=(num_samples, feature_dim))
        X = base_patterns + noise

        # Normalize features to roughly same scale as text embeddings
        X = (X - X.mean(axis=1, keepdims=True)) / (X.std(axis=1, keepdims=True) + 1e-9)

        # Produce targets as lower-dimensional spectral-like vectors
        Y = np.tanh(np.dot(X, np.random.randn(feature_dim, target_dim) * 0.02) + np.random.randn(num_samples, target_dim) * 0.05)

        X_tensor = torch.tensor(X, dtype=torch.float32)
        Y_tensor = torch.tensor(Y, dtype=torch.float32)
        return X_tensor, Y_tensor

    def save_sample(self, X: torch.Tensor, Y: torch.Tensor, path: str = "satellite_samples.pt") -> None:
        torch.save({'X': X, 'Y': Y}, path)


if __name__ == "__main__":
    # Quick demo when run standalone
    conn = SatelliteConnector(simulate=True)
    Xs, Ys = conn.fetch_latest_features(num_samples=10)
    print("Simulated satellite features:", Xs.shape, Ys.shape)
