"""2-layer stacked LSTM for crop yield prediction (PyTorch)."""

import torch
import torch.nn as nn


class CropYieldLSTM(nn.Module):
    def __init__(
        self,
        input_size:  int,
        hidden_size: int = 128,
        num_layers:  int = 2,
        dropout:     float = 0.2,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,   # applied between layers (only when num_layers > 1)
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_size)
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)   # (batch,)
