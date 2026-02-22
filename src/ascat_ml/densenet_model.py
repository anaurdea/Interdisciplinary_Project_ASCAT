from __future__ import annotations

import torch
from torch import nn


class DenseLayer(nn.Module):
    def __init__(self, in_channels: int, growth_rate: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(
            in_channels,
            growth_rate,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(self.relu(self.norm(x)))
        out = self.dropout(out)
        return torch.cat([x, out], dim=1)


class DenseBlock(nn.Module):
    def __init__(self, in_channels: int, n_layers: int, growth_rate: int, dropout: float) -> None:
        super().__init__()
        layers = []
        channels = in_channels
        for _ in range(n_layers):
            layers.append(DenseLayer(channels, growth_rate=growth_rate, dropout=dropout))
            channels += growth_rate
        self.layers = nn.ModuleList(layers)
        self.out_channels = channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for layer in self.layers:
            out = layer(out)
        return out


class Transition(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DenseNetRegressor(nn.Module):
    def __init__(
        self,
        in_channels: int,
        aux_dim: int = 0,
        *,
        growth_rate: int = 16,
        block_layers: tuple[int, int] = (4, 4),
        init_features: int = 32,
        dropout: float = 0.1,
        head_hidden: int = 64,
    ) -> None:
        super().__init__()
        self.aux_dim = int(aux_dim)

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, init_features, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(init_features),
            nn.ReLU(inplace=True),
        )

        channels = init_features
        self.block1 = DenseBlock(channels, n_layers=block_layers[0], growth_rate=growth_rate, dropout=dropout)
        channels = self.block1.out_channels
        self.trans1 = Transition(channels, out_channels=max(channels // 2, growth_rate))
        channels = max(channels // 2, growth_rate)

        self.block2 = DenseBlock(channels, n_layers=block_layers[1], growth_rate=growth_rate, dropout=dropout)
        channels = self.block2.out_channels
        self.norm_final = nn.BatchNorm2d(channels)
        self.relu_final = nn.ReLU(inplace=True)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        head_in = channels + self.aux_dim
        self.head = nn.Sequential(
            nn.Linear(head_in, head_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, 1),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x_img: torch.Tensor, x_aux: torch.Tensor | None = None) -> torch.Tensor:
        out = self.stem(x_img)
        out = self.block1(out)
        out = self.trans1(out)
        out = self.block2(out)
        out = self.relu_final(self.norm_final(out))
        out = self.pool(out).flatten(1)

        if self.aux_dim > 0:
            if x_aux is None:
                raise ValueError('x_aux is required when aux_dim > 0')
            out = torch.cat([out, x_aux], dim=1)

        return self.head(out).squeeze(-1)
