from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.preprocessing import StandardScaler


class AutoencoderNetwork(nn.Module):

    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x):
        compressed = self.encoder(x)
        reconstructed = self.decoder(compressed)
        return reconstructed


class WalletAutoencoder:

    def __init__(
        self,
        hidden_dim=16,
        latent_dim=8,
        epochs=50,
        learning_rate=0.001,
        random_state=42
    ):
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.random_state = random_state

        self.device = torch.device("cpu")

        self.scaler = StandardScaler()
        self.model = None
        self.feature_columns = None
        self.threshold = None
        self.is_fitted = False

    def _prepare_features(self, feature_matrix, fitting=False):

        if "wallet" not in feature_matrix.columns:
            raise ValueError("Feature matrix must contain a 'wallet' column.")

        if feature_matrix["wallet"].isna().any():
            raise ValueError("Wallet column cannot contain missing values.")

        if feature_matrix["wallet"].duplicated().any():
            raise ValueError("Each wallet must appear only once.")

        if fitting:
            numeric_features = feature_matrix.select_dtypes(
                include="number"
            )

            if numeric_features.empty:
                raise ValueError("No numeric features available.")

            self.feature_columns = numeric_features.columns.tolist()

        else:
            missing_columns = set(self.feature_columns) - set(
                feature_matrix.columns
            )

            if missing_columns:
                raise ValueError(
                    f"Missing model features: {sorted(missing_columns)}"
                )

            numeric_features = feature_matrix[self.feature_columns]

        numeric_features = numeric_features.replace(
            [np.inf, -np.inf],
            np.nan
        )

        numeric_features = numeric_features.fillna(
            numeric_features.median()
        ).fillna(0)

        values = numeric_features.to_numpy(dtype=np.float32)

        if not np.isfinite(values).all():
            raise ValueError("Features contain invalid numeric values.")

        return values

    def fit(self, feature_matrix):

        if len(feature_matrix) < 2:
            raise ValueError("At least two wallets are required for training.")

        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        values = self._prepare_features(
            feature_matrix,
            fitting=True
        )

        scaled_values = self.scaler.fit_transform(values)

        input_dim = scaled_values.shape[1]

        actual_hidden_dim = max(
            2,
            min(self.hidden_dim, input_dim * 2)
        )

        actual_latent_dim = max(
            1,
            min(self.latent_dim, input_dim // 2)
        )

        self.model = AutoencoderNetwork(
            input_dim=input_dim,
            hidden_dim=actual_hidden_dim,
            latent_dim=actual_latent_dim
        ).to(self.device)

        tensor_data = torch.tensor(
            scaled_values,
            dtype=torch.float32
        ).to(self.device)

        criterion = nn.MSELoss()

        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate
        )

        self.model.train()

        for _ in range(self.epochs):

            reconstructed = self.model(tensor_data)

            loss = criterion(reconstructed, tensor_data)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        self.is_fitted = True

        training_errors = self._calculate_errors(scaled_values)

        # The 95th percentile of training errors is our initial threshold.
        self.threshold = float(np.percentile(training_errors, 95))

        return self

    def _calculate_errors(self, scaled_values):

        self.model.eval()

        with torch.no_grad():

            tensor_data = torch.tensor(
                scaled_values,
                dtype=torch.float32
            ).to(self.device)

            reconstructed = self.model(tensor_data)

            errors = torch.mean(
                (tensor_data - reconstructed) ** 2,
                dim=1
            )

        return errors.cpu().numpy()

    def predict(self, feature_matrix):

        if not self.is_fitted:
            raise RuntimeError("Train the Autoencoder before prediction.")

        values = self._prepare_features(feature_matrix)

        scaled_values = self.scaler.transform(values)

        reconstruction_errors = self._calculate_errors(scaled_values)

        results = pd.DataFrame({
            "wallet": feature_matrix["wallet"].values,
            "reconstruction_error": reconstruction_errors,
            "is_anomaly": reconstruction_errors > self.threshold
        })

        results["prediction"] = np.where(
            results["is_anomaly"],
            "anomaly",
            "normal"
        )

        return results

    def save(self, path):

        if not self.is_fitted:
            raise RuntimeError("Cannot save an untrained Autoencoder.")

        model_path = Path(path)
        model_path.parent.mkdir(parents=True, exist_ok=True)

        torch.save(
            self.model.state_dict(),
            model_path
        )

        metadata = {
            "hidden_dim": self.hidden_dim,
            "latent_dim": self.latent_dim,
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "random_state": self.random_state,
            "feature_columns": self.feature_columns,
            "threshold": self.threshold,
            "scaler": self.scaler,
            "input_dim": len(self.feature_columns),
            "actual_hidden_dim": self.model.encoder[0].out_features,
            "actual_latent_dim": self.model.encoder[2].out_features
        }

        metadata_path = model_path.with_suffix(
            model_path.suffix + ".metadata"
        )

        joblib.dump(metadata, metadata_path)

    @classmethod
    def load(cls, path):

        model_path = Path(path)

        metadata_path = model_path.with_suffix(
            model_path.suffix + ".metadata"
        )

        metadata = joblib.load(metadata_path)

        instance = cls(
            hidden_dim=metadata["hidden_dim"],
            latent_dim=metadata["latent_dim"],
            epochs=metadata["epochs"],
            learning_rate=metadata["learning_rate"],
            random_state=metadata["random_state"]
        )

        instance.feature_columns = metadata["feature_columns"]
        instance.threshold = metadata["threshold"]
        instance.scaler = metadata["scaler"]

        instance.model = AutoencoderNetwork(
            input_dim=metadata["input_dim"],
            hidden_dim=metadata["actual_hidden_dim"],
            latent_dim=metadata["actual_latent_dim"]
        )

        state_dict = torch.load(
            model_path,
            map_location="cpu",
            weights_only=True
        )

        instance.model.load_state_dict(state_dict)
        instance.model.eval()

        instance.is_fitted = True

        return instance