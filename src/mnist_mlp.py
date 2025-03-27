import torch
import torch.nn as nn


class MLP(nn.Module):

    def __init__(
        self, fan_in: int, fan_out: int, dim_hidden_layers: list[int], activation: str
    ) -> None:
        super(MLP, self).__init__()
        self.l_in = nn.Linear(fan_in, dim_hidden_layers[0])
        self.ls_hidden = nn.ModuleList(
            [
                nn.Linear(dim_hidden_layers[i], dim_hidden_layers[i + 1])
                for i in range(len(dim_hidden_layers) - 1)
            ]
        )
        self.l_out = nn.Linear(dim_hidden_layers[-1], fan_out)
        if activation == "tanh":
            self.activation = nn.Tanh()
        else:
            self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.activation(self.l_in(x))
        for _, l_hidden in enumerate(self.ls_hidden):
            x = self.activation(l_hidden(x))
        out = self.activation(self.l_out(x))
        return out

    def param_count(self) -> int:
        count = sum(p.numel() for p in self.parameters())
        return count


class MLP_NO_OUTPUT_ACTIVATION(nn.Module):
    # try leakyReLU if no improvement
    def __init__(
        self, fan_in: int, fan_out: int, dim_hidden_layers: list[int], activation: str
    ) -> None:
        super(MLP_NO_OUTPUT_ACTIVATION, self).__init__()
        self.l_in = nn.Linear(fan_in, dim_hidden_layers[0])
        self.ls_hidden = nn.ModuleList(
            [
                nn.Linear(dim_hidden_layers[i], dim_hidden_layers[i + 1])
                for i in range(len(dim_hidden_layers) - 1)
            ]
        )
        self.l_out = nn.Linear(dim_hidden_layers[-1], fan_out)
        if activation == "tanh":
            self.activation = nn.Tanh()
        else:
            self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.activation(self.l_in(x))
        for _, l_hidden in enumerate(self.ls_hidden):
            x = self.activation(l_hidden(x))
        out = self.l_out(x)
        return out

    def param_count(self) -> int:
        count = sum(p.numel() for p in self.parameters())
        return count
