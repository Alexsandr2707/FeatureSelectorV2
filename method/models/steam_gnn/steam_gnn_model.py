import torch
import torch.nn as nn
import torch.nn.functional as F


class GRU_state(nn.GRU):
    def __init__(self, *args, last_state=True, residual=False, **argv):
        super().__init__(*args, **argv)
        self.last_state = last_state
        self.residual = residual

    def forward(self, x):  # type: ignore
        y = super().forward(x)[0]
        if self.residual:
            y = y + x
        if self.last_state:
            y = y[:, -1]
        return y


class GLU(nn.Module):
    def __init__(self, input_channel, output_channel):
        super(GLU, self).__init__()
        self.linear_left = nn.Linear(input_channel, output_channel)
        self.linear_right = nn.Linear(input_channel, output_channel)

    def forward(self, x):
        return torch.mul(self.linear_left(x), torch.sigmoid(self.linear_right(x)))


class StockBlockLayer(nn.Module):
    def __init__(self, time_step, unit, multi_layer, mask=None, stack_cnt=0):
        super(StockBlockLayer, self).__init__()
        # if mask is not None:
        #    self.add_input = nn.Sequential(
        #        nn.Linear(unit, 64),
        #        nn.LeakyReLU(0.2),
        #        nn.Linear(64, sum(mask)),
        #    )
        self.time_step = time_step
        self.mask = mask
        # self.unit = unit
        self.stack_cnt = stack_cnt
        self.multi = multi_layer
        self.weight = nn.Parameter(
            torch.Tensor(
                1, 3 + 1, 1, self.time_step * self.multi, self.multi * self.time_step
            )
        )  # [K+1, 1, in_c, out_c]
        nn.init.xavier_normal_(self.weight)
        self.forecast = nn.Linear(
            self.time_step * self.multi, self.time_step * self.multi
        )
        self.forecast_result = nn.Linear(self.time_step * self.multi, self.time_step)
        # if self.stack_cnt == 0:
        self.backcast = nn.Linear(self.time_step * self.multi, self.time_step)
        self.backcast_short_cut = nn.Linear(self.time_step, self.time_step)
        self.relu = nn.ReLU()
        self.GLUs = nn.ModuleList()
        self.output_channel = 4 * self.multi
        for i in range(3):
            if i == 0:
                self.GLUs.append(
                    GLU(self.time_step * 4, self.time_step * self.output_channel)
                )
                self.GLUs.append(
                    GLU(self.time_step * 4, self.time_step * self.output_channel)
                )
            elif i == 1:
                self.GLUs.append(
                    GLU(
                        self.time_step * self.output_channel,
                        self.time_step * self.output_channel,
                    )
                )
                self.GLUs.append(
                    GLU(
                        self.time_step * self.output_channel,
                        self.time_step * self.output_channel,
                    )
                )
            else:
                self.GLUs.append(
                    GLU(
                        self.time_step * self.output_channel,
                        self.time_step * self.output_channel,
                    )
                )
                self.GLUs.append(
                    GLU(
                        self.time_step * self.output_channel,
                        self.time_step * self.output_channel,
                    )
                )

    def spe_seq_cell(self, input):
        batch_size, k, input_channel, node_cnt, time_step = input.size()
        input = input.view(batch_size, -1, node_cnt, time_step)
        ffted = torch.view_as_real(torch.fft.fft(input, dim=1))
        real = (
            ffted[..., 0]
            .permute(0, 2, 1, 3)
            .contiguous()
            .reshape(batch_size, node_cnt, -1)
        )
        img = (
            ffted[..., 1]
            .permute(0, 2, 1, 3)
            .contiguous()
            .reshape(batch_size, node_cnt, -1)
        )
        for i in range(3):
            real = self.GLUs[i * 2](real)
            img = self.GLUs[2 * i + 1](img)
        real = (
            real.reshape(batch_size, node_cnt, 4, -1).permute(0, 2, 1, 3).contiguous()
        )
        img = img.reshape(batch_size, node_cnt, 4, -1).permute(0, 2, 1, 3).contiguous()
        time_step_as_inner = torch.cat([real.unsqueeze(-1), img.unsqueeze(-1)], dim=-1)
        iffted = torch.fft.irfft(
            torch.view_as_complex(time_step_as_inner),
            n=time_step_as_inner.shape[1],
            dim=1,
        )
        return iffted

    def forward(self, x, mul_L):
        mul_L = mul_L.unsqueeze(1)
        x = x.unsqueeze(1)
        gfted = torch.matmul(mul_L, x)
        # print("before", gfted.shape)
        if hasattr(self, "add_input"):
            xx = gfted
            xx = gfted.view(*(xx.shape[:-2] + xx.shape[-2:][::-1]))
            xx = torch.concat((xx, self.add_input(xx)), axis=-1)  # type: ignore
            xx = xx.view(*(xx.shape[:-2] + xx.shape[-2:][::-1]))
            gfted = xx
            # print("after", gfted.shape)
        gconv_input = self.spe_seq_cell(gfted).unsqueeze(2)
        # print("gconv_input", gconv_input.shape)
        # asd
        igfted = torch.matmul(gconv_input, self.weight)
        igfted = torch.sum(igfted, dim=1)
        # print("igfted", igfted.shape)
        forecast_source = torch.sigmoid(self.forecast(igfted).squeeze(1))
        # print("forecast_source", forecast_source.shape)
        forecast = self.forecast_result(forecast_source)
        # print("forecast", forecast.shape)
        # asd
        # if True:
        backcast_short = self.backcast_short_cut(x).squeeze(1)
        backcast_igifted = self.backcast(igfted)
        backcast_source = torch.sigmoid(backcast_igifted - backcast_short)
        # else:
        #    backcast_source = None
        if self.stack_cnt and self.mask is not None:
            backcast_source = backcast_source[..., ~self.mask, :]  # type:ignore
        return forecast, backcast_source


class Model(nn.Module):
    def __init__(
        self,
        units,
        stack_cnt,
        time_step,
        multi_layer,
        horizon=1,
        dropout_rate=0.5,
        leaky_rate=0.2,
        device="cpu",
        mask=None,
    ):
        torch.nn.functional.conv2d()
        super().__init__()
        self.unit = units
        if mask is None:
            pass
        else:
            self.add_input = nn.Sequential(
                # GRU_state(self.unit-1, 64, last_state=False),
                # nn.Linear(self.unit-1, 8, bias=True),
                # nn.LeakyReLU(0.2),
                # nn.Dropout(0.25),
                # nn.Linear(8, sum(mask), bias=True),
                nn.Linear(self.unit - 1, sum(mask), bias=False),
            )
        self.stack_cnt = stack_cnt
        # self.unit = units
        self.alpha = leaky_rate
        self.time_step = time_step
        self.horizon = horizon
        self.weight_key = nn.Parameter(torch.zeros(size=(self.unit, 1)))
        nn.init.xavier_uniform_(self.weight_key.data, gain=1.414)
        self.weight_query = nn.Parameter(torch.zeros(size=(self.unit, 1)))
        nn.init.xavier_uniform_(self.weight_query.data, gain=1.414)
        self.GRU = nn.GRU(self.time_step, self.unit)
        self.multi_layer = multi_layer
        self.stock_block = nn.ModuleList()
        self.stock_block.extend(
            [
                StockBlockLayer(
                    self.time_step,
                    self.unit,
                    self.multi_layer,
                    mask=mask,
                    stack_cnt=i == self.stack_cnt - 1,
                )
                for i in range(self.stack_cnt)
            ]
        )
        # mask=None, stack_cnt=i) for i in range(self.stack_cnt)])
        self.fc = nn.Sequential(
            nn.Linear(int(self.time_step), int(self.time_step)),
            nn.LeakyReLU(),
            nn.Linear(int(self.time_step), self.horizon),
        )
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.mask = mask
        self.to(device)

    def get_laplacian(self, graph, normalize):
        """
        return the laplacian of the graph.
        :param graph: the graph structure without self loop, [N, N].
        :param normalize: whether to used the normalized laplacian.
        :return: graph laplacian.
        """
        if normalize:
            D = torch.diag(torch.sum(graph, dim=-1) ** (-1 / 2))
            L = torch.eye(
                graph.size(0), device=graph.device, dtype=graph.dtype
            ) - torch.mm(torch.mm(D, graph), D)
        else:
            D = torch.diag(torch.sum(graph, dim=-1))
            L = D - graph
        return L

    def cheb_polynomial(self, laplacian):
        """
        Compute the Chebyshev Polynomial, according to the graph laplacian.
        :param laplacian: the graph laplacian, [N, N].
        :return: the multi order Chebyshev laplacian, [K, N, N].
        """
        N = laplacian.size(0)  # [N, N]
        laplacian = laplacian.unsqueeze(0)
        first_laplacian = torch.zeros(
            [1, N, N], device=laplacian.device, dtype=torch.float
        )
        second_laplacian = laplacian
        third_laplacian = (
            2 * torch.matmul(laplacian, second_laplacian)
        ) - first_laplacian
        forth_laplacian = (
            2 * torch.matmul(laplacian, third_laplacian) - second_laplacian
        )
        multi_order_laplacian = torch.cat(
            [first_laplacian, second_laplacian, third_laplacian, forth_laplacian], dim=0
        )
        return multi_order_laplacian

    def latent_correlation_layer(self, x):
        if hasattr(self, "add_input"):
            x = torch.concat((x, self.add_input(x)), axis=-1)  # type:ignore
        input, _ = self.GRU(x.permute(2, 0, 1).contiguous())
        input = input.permute(1, 0, 2).contiguous()
        attention = self.self_graph_attention(input)
        attention = torch.mean(attention, dim=0)
        degree = torch.sum(attention, dim=1)
        # laplacian is sym or not
        attention = 0.5 * (attention + attention.T)
        degree_l = torch.diag(degree)
        diagonal_degree_hat = torch.diag(1 / (torch.sqrt(degree) + 1e-7))
        laplacian = torch.matmul(
            diagonal_degree_hat, torch.matmul(degree_l - attention, diagonal_degree_hat)
        )
        mul_L = self.cheb_polynomial(laplacian)
        return mul_L, attention, x

    def self_graph_attention(self, input):
        input = input.permute(0, 2, 1).contiguous()
        bat, N, fea = input.size()
        key = torch.matmul(input, self.weight_key)
        query = torch.matmul(input, self.weight_query)
        data = key.repeat(1, 1, N).view(bat, N * N, 1) + query.repeat(1, N, 1)
        data = data.squeeze(2)
        data = data.view(bat, N, -1)
        data = self.leakyrelu(data)
        attention = F.softmax(data, dim=2)
        attention = self.dropout(attention)
        return attention

    def graph_fft(self, input, eigenvectors):
        return torch.matmul(eigenvectors, input)

    def forward(self, x):
        mul_L, attention, x = self.latent_correlation_layer(x)
        X = x.unsqueeze(1).permute(0, 1, 3, 2).contiguous()
        result = []
        xs = []
        for stack_i in range(self.stack_cnt):
            forecast, X = self.stock_block[stack_i](X, mul_L)
            result.append(forecast)
            xs.append(X)
        forecast = sum(result)
        # X = sum(xs)
        X = xs[-1]
        forecast = self.fc(forecast)
        backcast = X.squeeze(1).permute(0, 2, 1)
        if forecast.size()[-1] == 1:
            return forecast.unsqueeze(1).squeeze(-1), backcast, attention
        else:
            return forecast.permute(0, 2, 1).contiguous(), backcast, attention


import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


class TimeSeriesDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def _make_windows(
    X: np.ndarray,
    y: np.ndarray,
    time_step: int,
    horizon: int,
):
    if X.ndim != 2 or y.ndim != 2:
        raise ValueError("X and y must be two-dimensional arrays")
    if len(X) != len(y):
        raise ValueError("X and y must have the same number of rows")
    if len(X) < time_step + horizon:
        raise ValueError("Not enough rows to create a time-series window")

    windows_X = []
    windows_y = []
    for end in range(time_step, len(X) - horizon + 1):
        windows_X.append(X[end - time_step : end])
        windows_y.append(y[end : end + horizon].T)
    return np.asarray(windows_X), np.asarray(windows_y)


def create_dataloaders(
    train_X: np.ndarray,
    train_y: np.ndarray,
    test_X: np.ndarray,
    test_y: np.ndarray,
    time_step: int = 48,
    horizon: int = 1,
    batch_size: int = 32,
    val_ratio: float = 0.15,
):
    train_X, train_y = _make_windows(train_X, train_y, time_step, horizon)
    test_X, test_y = _make_windows(test_X, test_y, time_step, horizon)

    val_size = int(len(train_X) * val_ratio)
    if val_size:
        train_X_part, train_y_part = train_X[:-val_size], train_y[:-val_size]
        val_X, val_y = train_X[-val_size:], train_y[-val_size:]
    else:
        train_X_part, train_y_part = train_X, train_y
        val_X = val_y = None

    train_loader = DataLoader(
        TimeSeriesDataset(train_X_part, train_y_part),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )
    test_loader = DataLoader(
        TimeSeriesDataset(test_X, test_y), batch_size=batch_size, shuffle=False
    )
    val_loader = (
        DataLoader(
            TimeSeriesDataset(val_X, val_y), batch_size=batch_size, shuffle=False
        )
        if val_X is not None
        else None
    )
    return train_loader, val_loader, test_loader


def train_model(
    model: Model,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    epochs: int = 50,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: str = "cpu",
    patience: int = 10,
):
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=8, min_lr=1e-5
    )
    criterion = nn.MSELoss()
    best_state = None
    best_val_loss = float("inf")
    patience_counter = 0
    history = {"train_loss": [], "val_loss": []}

    for _ in range(epochs):
        model.train()
        train_losses = []
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            forecast, _, _ = model(X_batch)
            target = y_batch
            forecast = forecast[..., -1].unsqueeze(1)
            if forecast.shape != target.shape:
                raise ValueError(
                    f"Forecast shape {forecast.shape} does not match target shape {target.shape}"
                )
            loss = criterion(forecast, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        if val_loader is not None:
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    forecast, _, _ = model(X_batch.to(device))
                    target = y_batch.to(device)
                    forecast = forecast[..., -1].unsqueeze(1)
                    if forecast.shape != target.shape:
                        raise ValueError(
                            f"Forecast shape {forecast.shape} does not match target shape {target.shape}"
                        )
                    val_losses.append(criterion(forecast, target).item())
        current_val_loss = (
            float(np.mean(val_losses)) if val_losses else float(np.mean(train_losses))
        )
        history["train_loss"].append(float(np.mean(train_losses)))
        history["val_loss"].append(current_val_loss)
        scheduler.step(current_val_loss)

        if current_val_loss < best_val_loss:
            best_val_loss = current_val_loss
            best_state = {
                key: value.cpu().clone() for key, value in model.state_dict().items()
            }
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history
