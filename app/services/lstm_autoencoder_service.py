from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


SEQUENCE_LENGTH = 12
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.001
HIDDEN_SIZE = 32


class LSTMAutoencoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()

        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
        )

        self.decoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
        )

        self.output_layer = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        _, (hidden, _) = self.encoder(x)

        last_hidden = hidden[-1]

        repeated_hidden = last_hidden.unsqueeze(1).repeat(
            1,
            x.size(1),
            1,
        )

        decoded, _ = self.decoder(repeated_hidden)

        reconstructed = self.output_layer(decoded)

        return reconstructed


def create_sequences_per_sensor_stream(feature_df, feature_columns, sequence_length):
    all_sequences = []
    sequence_metadata = []

    grouped = feature_df.groupby(["device_id", "parameter"])

    for (device_id, parameter), group_df in grouped:
        group_df = group_df.sort_values("timestamp").copy()

        values = group_df[feature_columns].values

        if len(values) < sequence_length:
            print(f"Skipping {device_id} + {parameter}: not enough rows")
            continue

        for index in range(len(values) - sequence_length + 1):
            sequence = values[index : index + sequence_length]

            all_sequences.append(sequence)

            sequence_metadata.append(
                {
                    "device_id": device_id,
                    "parameter": parameter,
                    "start_timestamp": group_df.iloc[index]["timestamp"],
                    "end_timestamp": group_df.iloc[index + sequence_length - 1]["timestamp"],
                }
            )

    return np.array(all_sequences), sequence_metadata


def train_lstm_autoencoder(feature_df, feature_columns, model_folder: Path):
    scaler = StandardScaler()

    feature_df = feature_df.copy()

    feature_df[feature_columns] = scaler.fit_transform(
        feature_df[feature_columns]
    )

    sequences, sequence_metadata = create_sequences_per_sensor_stream(
        feature_df,
        feature_columns,
        SEQUENCE_LENGTH,
    )

    if len(sequences) == 0:
        raise ValueError("Not enough data to create LSTM sequences.")

    x_tensor = torch.tensor(sequences, dtype=torch.float32)

    dataset = TensorDataset(x_tensor)

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    input_size = x_tensor.shape[2]

    model = LSTMAutoencoder(
        input_size=input_size,
        hidden_size=HIDDEN_SIZE,
    )

    loss_function = nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    losses = []

    for epoch in range(EPOCHS):
        model.train()

        epoch_losses = []

        for batch in dataloader:
            batch_x = batch[0]

            reconstructed = model(batch_x)

            loss = loss_function(reconstructed, batch_x)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())

        average_loss = sum(epoch_losses) / len(epoch_losses)

        losses.append(average_loss)

        mlflow.log_metric(
            "training_loss",
            average_loss,
            step=epoch + 1,
        )

        print(f"Epoch {epoch + 1}/{EPOCHS} - Loss: {average_loss:.6f}")

    model.eval()

    with torch.no_grad():
        reconstructed = model(x_tensor)

        reconstruction_errors = torch.mean(
            (reconstructed - x_tensor) ** 2,
            dim=(1, 2),
        ).numpy()

    reconstruction_threshold = float(
        np.percentile(reconstruction_errors, 95)
    )

    model_folder.mkdir(parents=True, exist_ok=True)

    model_file = model_folder / "lstm_autoencoder.pt"
    scaler_file = model_folder / "lstm_autoencoder_scaler.joblib"
    metadata_file = model_folder / "lstm_autoencoder_sequence_metadata.csv"

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_size": input_size,
            "hidden_size": HIDDEN_SIZE,
            "sequence_length": SEQUENCE_LENGTH,
            "feature_columns": feature_columns,
            "reconstruction_threshold": reconstruction_threshold,
            "sequence_mode": "per_device_and_parameter",
        },
        model_file,
    )

    joblib.dump(scaler, scaler_file)

    metadata_df = pd.DataFrame(sequence_metadata)
    metadata_df.to_csv(metadata_file, index=False)

    return {
        "model_file": model_file,
        "scaler_file": scaler_file,
        "metadata_file": metadata_file,
        "training_rows": len(feature_df),
        "training_sequences": len(sequences),
        "final_training_loss": losses[-1],
        "reconstruction_threshold": reconstruction_threshold,
        "input_size": input_size,
    }