import torch
from torch import nn


class MovingAvg(nn.Module):
    def __init__(self, kernel=25):
        super().__init__()
        # slide on the last dimension
        self.avg = nn.AvgPool1d(kernel, stride=1, padding=0)
        self.kernel = kernel

    def forward(self, x):  # x: (batch_size, features_in, look_back) (B, C, L)
        front = x[:, :, :1].repeat(1, 1, (self.kernel - 1) // 2)
        end = x[:, :, -1:].repeat(1, 1, (self.kernel - 1) // 2)
        x = torch.cat([front, x, end], dim=2)  # [B, C, L']
        return self.avg(x)  # [B, C, L]


class Linear(nn.Module):
    """Direct map: ŷ = W x + b (no decomp / recenter).

    individual=False (shared W): one Linear(L→H) applied to every channel.
    individual=True (CI): one Linear per channel; needs n_vars.
    """

    def __init__(self, seq_len, pred_len, n_vars=None, individual=False):
        super().__init__()
        self.individual = individual
        if individual:
            if n_vars is None:
                raise ValueError("n_vars required when individual=True")
            self.Linear = nn.ModuleList(
                [nn.Linear(seq_len, pred_len) for _ in range(n_vars)]
            )
        else:
            self.Linear = nn.Linear(seq_len, pred_len)

    def forward(self, x):  # B C L -> B C H
        if self.individual:
            return torch.stack(
                [self.Linear[i](x[:, i, :]) for i in range(x.shape[1])], dim=1
            )
        return self.Linear(x)


class DLinear(nn.Module):
    """MA decomp + dual Linear heads (seasonal / trend).

    individual=False (shared W): one Linear(L→H) applied to every channel.
    individual=True (CI): one Linear per channel; needs n_vars.
    """

    def __init__(self, seq_len, pred_len, n_vars=None, kernel=3, individual=False):
        super().__init__()
        self.decomp = MovingAvg(kernel)
        self.individual = individual
        if individual:
            if n_vars is None:
                raise ValueError("n_vars required when individual=True")
            self.Linear_Seasonal = nn.ModuleList(
                [nn.Linear(seq_len, pred_len) for _ in range(n_vars)]
            )
            self.Linear_Trend = nn.ModuleList(
                [nn.Linear(seq_len, pred_len) for _ in range(n_vars)]
            )
        else:
            self.Linear_Seasonal = nn.Linear(seq_len, pred_len)
            self.Linear_Trend = nn.Linear(seq_len, pred_len)

    def forward(self, x):  # B C L -> B C H
        t = self.decomp(x)
        s = x - t
        if self.individual:
            s_out = torch.stack(
                [self.Linear_Seasonal[i](s[:, i, :]) for i in range(s.shape[1])], dim=1
            )
            t_out = torch.stack(
                [self.Linear_Trend[i](t[:, i, :]) for i in range(t.shape[1])], dim=1
            )
        else:
            s_out = self.Linear_Seasonal(s)
            t_out = self.Linear_Trend(t)
        return s_out + t_out  # B C H


class NLinear(nn.Module):
    """Last-value recenter: x' = x - l → Linear → ŷ = ŷ' + l.

    individual=False (shared W): one Linear(L→H) applied to every channel.
    individual=True (CI): one Linear per channel; needs n_vars.
    """

    def __init__(self, seq_len, pred_len, n_vars=None, individual=False):
        super().__init__()
        self.individual = individual
        if individual:
            if n_vars is None:
                raise ValueError("n_vars required when individual=True")
            self.Linear = nn.ModuleList(
                [nn.Linear(seq_len, pred_len) for _ in range(n_vars)]
            )
        else:
            self.Linear = nn.Linear(seq_len, pred_len)

    def forward(self, x):  # B C L -> B C H
        last = x[:, :, -1:]  # (B, C, 1); broadcast for sub / add-back
        x = x - last
        if self.individual:
            out = torch.stack(
                [self.Linear[i](x[:, i, :]) for i in range(x.shape[1])], dim=1
            )
        else:
            out = self.Linear(x)
        return out + last
