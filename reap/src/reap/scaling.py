import numpy as np


class TargetScaler:
    def __init__(self, mode="zscore", mean=0.0, std=1.0, min_=0.0, max_=1.0):
        self.mode = mode
        self.mean = float(mean)
        self.std = float(std)
        self.min_ = float(min_)
        self.max_ = float(max_)

    def fit(self, y):
        y = np.asarray(y, dtype=float)
        if self.mode == "zscore":
            self.mean = float(np.mean(y))
            self.std = float(np.std(y) or 1.0)
        elif self.mode == "minmax":
            self.min_ = float(np.min(y))
            self.max_ = float(np.max(y))
        return self

    def transform(self, y):
        y = np.asarray(y, dtype=float)
        if self.mode == "zscore":
            return (y - self.mean) / self.std
        if self.mode == "minmax":
            den = self.max_ - self.min_
            return (y - self.min_) / den if abs(den) > 1e-12 else y * 0.0
        if self.mode.startswith("log1p"):
            return np.log1p(y)
        return y

    def inverse(self, y):
        y = np.asarray(y, dtype=float)
        if self.mode == "zscore":
            return y * self.std + self.mean
        if self.mode == "minmax":
            return y * (self.max_ - self.min_) + self.min_
        if self.mode.startswith("log1p"):
            return np.expm1(y)
        return y

    def get_params(self):
        return {
            "mode": self.mode,
            "mean": self.mean,
            "std": self.std,
            "min_": self.min_,
            "max_": self.max_,
        }

    def set_params(self, params):
        for key, value in params.items():
            setattr(self, key, value)
