"""检测机制 3/3：Autoencoder（自编码器重构误差异常检测）。

思路（对齐 `papers/AUTO.pdf`、`Autoencoder-Based_Anomaly_Detection.pdf`）：
  只在**良性流量**上训练一个欠完备自编码器；正常样本能被很好地重构，
  攻击流量重构误差大。

      anomaly_score = mean_j (z_j - x_j)^2      (逐样本平均重构误差)

  * 阈值：良性重构误差的高分位数（默认 95%）/ 有标签时 F1 扫描；
  * embedding：逐维重构残差 (x - x_hat)，让 MADDPG 的 critic 知道
    **是哪些特征没被重构出来**——这对"区分攻击类型"很有用。

依赖 torch（环境里已装 torch 2.5.1）。
"""
from __future__ import annotations

import os
from typing import Any, List, Optional, Sequence

import numpy as np

from .base import DetectorBase, DetectorOutput, register_detector


@register_detector("autoencoder")
@register_detector("ae")
class AutoencoderDetector(DetectorBase):
    def __init__(
        self,
        hidden_dims: Sequence[int] = (64, 32),
        latent_dim: int = 8,
        epochs: int = 30,
        batch_size: int = 256,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        alpha: float = 0.05,             # 目标误报率
        threshold: Optional[float] = None,
        embed_dim: Optional[int] = None,  # None -> 用逐维残差，维度 = 输入特征数
        dropout: float = 0.0,
        device: Optional[str] = None,
        random_state: Optional[int] = 42,
        verbose: bool = False,
        **_: Any,
    ) -> None:
        self.hidden_dims: List[int] = [int(h) for h in hidden_dims]
        self.latent_dim = int(latent_dim)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.alpha = float(alpha)
        self.threshold = float(threshold) if threshold is not None else None
        self._embed_dim_cfg = embed_dim
        self.dropout = float(dropout)
        self.device = device
        self.random_state = random_state
        self.verbose = bool(verbose)

        self.model = None
        self.input_dim: Optional[int] = None
        self._mu = 0.0
        self._sd = 1.0
        self.embed_dim = 0

    # ------------------------------------------------------------------ 网络
    def _build(self, input_dim: int):
        import torch
        import torch.nn as nn

        layers: List[nn.Module] = []
        prev = input_dim
        for h in self.hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            if self.dropout > 0:
                layers.append(nn.Dropout(self.dropout))
            prev = h
        layers.append(nn.Linear(prev, self.latent_dim))
        encoder = nn.Sequential(*layers)

        dec_layers: List[nn.Module] = []
        prev = self.latent_dim
        for h in reversed(self.hidden_dims):
            dec_layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        dec_layers.append(nn.Linear(prev, input_dim))
        decoder = nn.Sequential(*dec_layers)

        class _AE(nn.Module):
            def __init__(self, enc: nn.Module, dec: nn.Module) -> None:
                super().__init__()
                self.encoder = enc
                self.decoder = dec

            def forward(self, x):  # noqa: D102
                z = self.encoder(x)
                return self.decoder(z)

        return _AE(encoder, decoder)

    def _resolve_device(self):
        import torch

        if self.device:
            return torch.device(self.device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------ fit
    def fit(
        self,
        X_normal: np.ndarray,
        X_test: Optional[np.ndarray] = None,
        y_test: Optional[np.ndarray] = None,
        **kwargs: Any,
    ) -> "AutoencoderDetector":
        import torch
        import torch.nn as nn

        X_normal = np.asarray(X_normal, dtype=np.float32)
        self.input_dim = int(X_normal.shape[1])
        self.embed_dim = (
            self.input_dim if self._embed_dim_cfg is None else int(self._embed_dim_cfg)
        )

        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)

        dev = self._resolve_device()
        self.model = self._build(self.input_dim).to(dev)
        opt = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        loss_fn = nn.MSELoss()

        tensor = torch.from_numpy(X_normal).to(dev)
        ds = torch.utils.data.TensorDataset(tensor)
        loader = torch.utils.data.DataLoader(
            ds, batch_size=self.batch_size, shuffle=True, drop_last=False
        )

        self.model.train()
        for ep in range(self.epochs):
            total, nb = 0.0, 0
            for (batch,) in loader:
                recon = self.model(batch)
                loss = loss_fn(recon, batch)
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += float(loss.item())
                nb += 1
            if self.verbose and (ep % max(1, self.epochs // 10) == 0 or ep == self.epochs - 1):
                print(f"  [AE] epoch {ep + 1}/{self.epochs} mse={total / max(nb, 1):.6f}")

        err_normal = self._recon_error(X_normal)
        self._mu = float(err_normal.mean())
        self._sd = float(err_normal.std() + 1e-8)

        if self.threshold is None:
            err_test = self._recon_error(np.asarray(X_test, dtype=np.float32)) if X_test is not None else None
            self.threshold = self._calibrate_threshold(
                err_normal, alpha=self.alpha, scores_test=err_test, y_test=y_test
            )
        return self

    # ------------------------------------------------------------------ score
    def _recon_error(self, X: np.ndarray) -> np.ndarray:
        resid = self._residuals(X)
        return (resid ** 2).mean(axis=1).astype(np.float32)

    def _residuals(self, X: np.ndarray) -> np.ndarray:
        import torch

        if self.model is None:
            raise RuntimeError("AutoencoderDetector 尚未 fit()")
        dev = next(self.model.parameters()).device
        self.model.eval()
        outs = []
        with torch.no_grad():
            for i in range(0, X.shape[0], 4096):
                batch = torch.from_numpy(np.asarray(X[i:i + 4096], dtype=np.float32)).to(dev)
                recon = self.model(batch)
                outs.append((batch - recon).cpu().numpy())
        return np.concatenate(outs, axis=0).astype(np.float32)

    def score_batch(self, X: np.ndarray) -> DetectorOutput:
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 1:
            X = X[None, :]
        resid = self._residuals(X)
        score = (resid ** 2).mean(axis=1).astype(np.float32)

        thr = float(self.threshold) if self.threshold is not None else float(self._mu + 3.0 * self._sd)
        flag = (score >= thr).astype(np.float32)

        z = (score - self._mu) / self._sd
        conf = 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))
        trust = np.where(flag > 0, conf, 1.0 - conf).astype(np.float32)

        emb = None
        if self.embed_dim > 0:
            emb = resid[:, : self.embed_dim].astype(np.float32)
            if emb.shape[1] < self.embed_dim:
                emb = np.concatenate(
                    [emb, np.zeros((emb.shape[0], self.embed_dim - emb.shape[1]), dtype=np.float32)],
                    axis=1,
                )
        return DetectorOutput(score=score, flag=flag, trust=trust, embedding=emb,
                              extra={"threshold": thr})

    def reset(self) -> None:
        return None  # 无时序状态

    # ------------------------------------------------------------------ 落盘
    def save(self, path: str) -> None:
        import torch

        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save(
            {
                "state_dict": self.model.state_dict() if self.model is not None else None,
                "input_dim": self.input_dim,
                "hidden_dims": self.hidden_dims,
                "latent_dim": self.latent_dim,
                "threshold": self.threshold,
                "mu": self._mu,
                "sd": self._sd,
                "embed_dim": self.embed_dim,
            },
            path,
        )

    def load(self, path: str) -> "AutoencoderDetector":
        import torch

        blob = torch.load(path, map_location=self._resolve_device(), weights_only=False)
        self.input_dim = int(blob["input_dim"])
        self.hidden_dims = list(blob["hidden_dims"])
        self.latent_dim = int(blob["latent_dim"])
        self.embed_dim = int(blob["embed_dim"])
        self.model = self._build(self.input_dim).to(self._resolve_device())
        self.model.load_state_dict(blob["state_dict"])
        self.model.eval()
        self.threshold = blob["threshold"]
        self._mu = float(blob["mu"])
        self._sd = float(blob["sd"])
        return self
