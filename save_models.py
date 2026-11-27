import pandas as pd
from backend.ml.models.train import train_models

# Sample wallet feature data
data = {
    "wallet": [f"wallet_{i}" for i in range(1, 21)],
    "fan_in": [2, 3, 4, 2, 5, 3, 4, 2, 3, 5,
               4, 3, 2, 5, 4, 3, 2, 4, 5, 3],
    "fan_out": [3, 2, 5, 4, 2, 3, 4, 5, 2, 3,
                4, 2, 5, 3, 4, 2, 3, 5, 4, 2],
    "transaction_frequency": [5, 8, 6, 4, 9, 7, 5, 6, 8, 4,
                              7, 5, 6, 9, 4, 8, 5, 7, 6, 4],
    "average_amount": [100, 200, 150, 300, 120, 180, 250, 130, 220, 170,
                       190, 140, 260, 110, 230, 160, 210, 125, 280, 175]
}

feature_matrix = pd.DataFrame(data)

# Train and save both models
train_models(
    feature_matrix,
    output_dir="backend/ml/models/saved",
    autoencoder_epochs=10
)

print("\nModels saved successfully!")