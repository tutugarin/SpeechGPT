import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class FCAdapter(nn.Module):
    def __init__(self, m=5, d_x=1280, fc_layer_dim=11264, llm_hidden_size=896):
        super().__init__()
        self.m = m
        self.conv = nn.Conv1d(
            in_channels=d_x,
            out_channels=d_x,
            kernel_size=m,
            stride=m,
            groups=d_x
        )

        self.linear1 = nn.Linear(d_x, fc_layer_dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(fc_layer_dim, llm_hidden_size)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = x.permute(0, 2, 1)

        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x


class TransformerAdapter(nn.Module):
    def __init__(self, d_x=1280, transform_dim=1024, num_heads=8, ff_dim=2048,
                 num_layers=2, dropout=0.1, llm_hidden_size=896):
        super().__init__()
        self.input_proj = nn.Linear(d_x, transform_dim)

        self.encoder_layer = TransformerEncoderLayer(
            d_model=transform_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout
        )
        self.transformer_encoder = TransformerEncoder(
            self.encoder_layer,
            num_layers=num_layers
        )

        self.linear1 = nn.Linear(transform_dim, ff_dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(ff_dim, llm_hidden_size)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.transformer_encoder(x)

        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x