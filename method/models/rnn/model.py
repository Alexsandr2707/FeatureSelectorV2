# type: ignore
import os
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
from datetime import datetime

import logging

logger = logging.getLogger(__name__)
# from IPython.display import clear_output

torch.manual_seed(0)
PATH = "%s.tmp"


class Repeat(nn.Module):
    def __init__(self, lag):
        super().__init__()
        self.lag = lag

    def forward(self, x):
        x = x.view(-1, 1, x.size()[-1])
        x = x.repeat(1, self.lag, 1)
        return x


class GRU_state(nn.GRU):
    def __init__(self, *args, last_state=True, residual=False, **argv):
        super().__init__(*args, **argv)
        self.last_state = last_state
        self.residual = residual

    def forward(self, input, hx=None):
        y = super().forward(input, hx)[0]
        if self.residual:
            y = y + input
        if self.last_state:
            y = y[:, -1]
        return y


class Reshape(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(self.shape)


class BaseData(torch.utils.data.Dataset):
    def __init__(self, device, *args):
        self.device = device
        self.args = args
        assert len(np.unique([len(x) for x in args])) == 1

    def __len__(self):
        return len(self.args[0])

    def __getitem__(self, idx):
        return [
            torch.Tensor(x[idx].to(torch.float32)).to(self.device) for x in self.args
        ]


class BaseModel(nn.Module):
    def count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def iterate(self, data, func, **kwargs):
        store = []
        for portion in data:
            store.append(func(*portion, **kwargs))
        return store

    def batch(self, X, y, penalty_func=None):
        self.optimizer.zero_grad()
        pred = self(X)
        loss = self.loss(pred, y)

        if penalty_func is not None:
            loss = loss + penalty_func()

        loss.backward()

        # torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=5.0)

        item = loss.item()

        if not np.isnan(item):
            self.optimizer.step()

        return pred.detach(), item

    def epoch(self, data, penalty_func=None):
        self.train()
        running_loss = 0
        counter = 0
        for portion in data:
            _, loss = self.batch(*portion, penalty_func=penalty_func)
            if not np.isnan(loss):
                running_loss += loss
                counter += 1
        assert counter != 0, "Model has gone NaN! - train"
        return running_loss / counter

    def valid(self, data):
        self.eval()
        running_loss = 0
        counter = 0
        with torch.no_grad():
            for portion in data:
                X, y = portion
                loss = self.loss(self(X), y).item()
                if not np.isnan(loss):
                    running_loss += loss
                    counter += 1
        assert counter != 0, "Model has gone NaN! - valid"
        return running_loss / counter

    def fit(
        self,
        train,
        valid=None,
        epochs=10,
        early_stopping_rounds=3,
        restore=True,
        verbose=True,
        penalty_func=None,
    ):
        best_valid = float("inf")
        counter = 0
        path = PATH % self.__class__.__name__

        for epoch in range(epochs):
            if counter > early_stopping_rounds:
                break

            train_ = self.epoch(
                train() if callable(train) else train, penalty_func=penalty_func
            )
            if valid is None:
                continue

            valid_ = self.valid(valid() if callable(valid) else valid)

            if valid_ < best_valid:
                counter = 0
                best_valid = valid_
                torch.save(self.state_dict(), path)
            else:
                counter += 1

            if hasattr(self, "scheduler"):
                self.scheduler.step()

            if verbose:
                lr = self.optimizer.param_groups[0]["lr"]
                logger.debug(
                    f"epoch {epoch}: train: {train_:1.4f}, lr: {lr:.3e}, valid: {valid_:1.4f}, best_valid: {best_valid:1.4f}"
                )

        if valid and restore:
            self.load_state_dict(torch.load(path))

    def true_pred(self, X, y):
        return y.cpu().numpy(), self(X).cpu().detach().numpy()

    def predict(self, data, **kwargs):
        self.eval()
        with torch.no_grad():
            result = self.iterate(data, self.true_pred, **kwargs)
        return np.concatenate([x[0] for x in result]), np.concatenate(
            [x[1] for x in result]
        )


class BaseModelEMA(BaseModel):
    def __init__(self, ema_decay=0.999, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ema_decay = ema_decay
        self.ema_shadow = {}
        self.ema_backup = {}
        self.ema_initialized = False

    def _init_ema(self):
        for name, param in self.named_parameters():
            if param.requires_grad:
                self.ema_shadow[name] = param.data.clone().detach()
        self.ema_initialized = True

    def _update_ema(self):
        if not self.ema_initialized:
            self._init_ema()
            return

        with torch.no_grad():
            for name, param in self.named_parameters():
                if param.requires_grad:
                    # shadow = decay * shadow + (1 - decay) * param
                    self.ema_shadow[name].sub_(
                        (1.0 - self.ema_decay) * (self.ema_shadow[name] - param.data)
                    )

    def apply_ema(self):
        if not self.ema_initialized:
            return

        self.ema_backup = {}
        with torch.no_grad():
            for name, param in self.named_parameters():
                if param.requires_grad:
                    self.ema_backup[name] = param.data.clone().detach()
                    param.data.copy_(self.ema_shadow[name])

    def restore_ema(self):
        if not self.ema_backup:
            return

        with torch.no_grad():
            for name, param in self.named_parameters():
                if param.requires_grad:
                    param.data.copy_(self.ema_backup[name])
        self.ema_backup = {}

    def load_state_dict(self, *args, **kwargs):
        res = super().load_state_dict(*args, **kwargs)
        self._init_ema()
        return res

    def batch(self, X, y, penalty_func=None):
        pred, item = super().batch(X, y, penalty_func=penalty_func)
        if not np.isnan(item):
            self._update_ema()
        return pred, item

    def fit(
        self,
        train,
        valid=None,
        epochs=10,
        early_stopping_rounds=3,
        restore=True,
        clear=None,
        verbose=True,
        penalty_func=None,
    ):
        best_valid = float("inf")
        counter = 0
        path = PATH % self.__class__.__name__

        for epoch in range(epochs):
            if counter > early_stopping_rounds:
                break
            if verbose and (clear is not None) and (epoch % clear == 0):
                pass

            train_ = self.epoch(
                train() if callable(train) else train, penalty_func=penalty_func
            )

            if valid is None:
                continue

            self.apply_ema()

            valid_ = self.valid(valid() if callable(valid) else valid)

            if valid_ < best_valid:
                counter = 0
                best_valid = valid_
                torch.save(self.state_dict(), path)
            else:
                counter += 1

            self.restore_ema()

            if hasattr(self, "scheduler"):
                self.scheduler.step()

            if verbose:
                lr = self.optimizer.param_groups[0]["lr"]
                logger.debug(
                    f"epoch {epoch}: train: {train_:1.4f}, lr: {lr:.3e}, valid: {valid_:1.4f}, best_valid: {best_valid:1.4f}"
                )

        if valid and restore:
            self.load_state_dict(torch.load(path))

    def predict(self, data, **kwargs):
        self.apply_ema()
        res = super().predict(data, **kwargs)
        self.restore_ema()
        return res
