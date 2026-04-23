"""
model.py — MLP definition, training routines, and inference helpers.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class MLP(nn.Module):
    def __init__(self, input_dim=30, hidden=64, dropout_p=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(hidden // 2, 2),
        )

    def forward(self, x):
        return self.net(x)


def make_mlp(seed, input_dim=30, hidden=64, dropout_p=0.3):
    """Instantiate MLP with deterministic initialization."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return MLP(input_dim=input_dim, hidden=hidden, dropout_p=dropout_p)


def train_model(model, X_np, y_np, n_epochs=50, lr=1e-3, batch_size=16, seed=None):
    """Standard cross-entropy training on numpy arrays."""
    if seed is not None:
        torch.manual_seed(seed)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    X_t = torch.tensor(X_np, dtype=torch.float32)
    y_t = torch.tensor(y_np, dtype=torch.long)

    # Create mini-batches for training
    loader = DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=True)

    model.train()
    for _ in range(n_epochs):
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()


def train_model_weighted(
    model, X_np, y_np, weights_np=None,
    n_epochs=50, lr=1e-3, batch_size=16, seed=None
):
    """Training with per-sample entropy weights."""
    if seed is not None:
        torch.manual_seed(seed)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(reduction="none")

    if weights_np is None:
        weights_np = np.ones(len(y_np))

    X_t = torch.tensor(X_np, dtype=torch.float32)
    y_t = torch.tensor(y_np, dtype=torch.long)
    w_t = torch.tensor(weights_np, dtype=torch.float32)

    # Each batch includes inputs, labels, and sample weights
    loader = DataLoader(TensorDataset(X_t, y_t, w_t), batch_size=batch_size, shuffle=True)

    model.train()
    for _ in range(n_epochs):
        for xb, yb, wb in loader:
            optimizer.zero_grad()
            logits = model(xb)
            unweighted_loss = criterion(logits, yb)

            # Compute weighted average loss across the batch
            weighted_loss = (unweighted_loss * wb).sum() / (wb.sum() + 1e-8)

            weighted_loss.backward()
            optimizer.step()


def train_model_cost_sensitive(
    model, X_np, y_np,
    class_weights=(1.0, 2.5),
    n_epochs=50, lr=1e-3, batch_size=16, seed=None
):
    """Training with an asymmetric class penalty (cost-sensitive)."""
    if seed is not None:
        torch.manual_seed(seed)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Apply higher penalty to the positive class if desired
    weights_t = torch.tensor(list(class_weights), dtype=torch.float32)
    criterion = nn.CrossEntropyLoss(weight=weights_t)

    X_t = torch.tensor(X_np, dtype=torch.float32)
    y_t = torch.tensor(y_np, dtype=torch.long)

    loader = DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=True)

    model.train()
    for _ in range(n_epochs):
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()


def train_model_joint_weighted(
    model, X_np, y_np,
    sample_weights_np=None,
    class_weights=(1.0, 1.0),
    n_epochs=50, lr=1e-3, batch_size=16, seed=None
):
    """Combined cost-sensitive + per-sample entropy weighting."""
    if seed is not None:
        torch.manual_seed(seed)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    class_w_t = torch.tensor(list(class_weights), dtype=torch.float32)
    criterion = nn.CrossEntropyLoss(weight=class_w_t, reduction="none")

    if sample_weights_np is None:
        sample_weights_np = np.ones(len(y_np), dtype=np.float32)

    X_t = torch.tensor(X_np, dtype=torch.float32)
    y_t = torch.tensor(y_np, dtype=torch.long)
    w_t = torch.tensor(sample_weights_np, dtype=torch.float32)

    # Each batch includes inputs, labels, and sample weights
    loader = DataLoader(TensorDataset(X_t, y_t, w_t), batch_size=batch_size, shuffle=True)

    model.train()
    for _ in range(n_epochs):
        for xb, yb, wb in loader:
            optimizer.zero_grad()
            logits = model(xb)
            losses = criterion(logits, yb)

            # Combine class-weighted loss with sample-level weighting
            loss = (losses * wb).sum() / (wb.sum() + 1e-8)

            loss.backward()
            optimizer.step()


def predict_proba_nn(model, X_np):
    """Return softmax class probabilities, shape (n, 2)."""
    model.eval()
    X_t = torch.tensor(X_np, dtype=torch.float32)

    with torch.no_grad():
        probs = torch.softmax(model(X_t), dim=1).cpu().numpy()

    return probs


def mc_dropout_probs(model, X_np, T=30, seed=None):
    """
    T stochastic forward passes with dropout enabled (MC-Dropout).
    Returns array of shape (T, n, 2).
    """
    if seed is not None:
        torch.manual_seed(seed)

    model.train()  # Keep dropout active during inference
    X_t = torch.tensor(X_np, dtype=torch.float32)
    out = np.zeros((T, X_np.shape[0], 2), dtype=np.float32)

    with torch.no_grad():
        for t in range(T):
            logits = model(X_t)
            out[t] = torch.softmax(logits, dim=1).cpu().numpy()

    model.eval()
    return out


def predictive_entropy(probs, eps=1e-12):
    """Row-wise predictive entropy in nats. probs shape: (n, K)."""
    p = np.clip(probs, eps, 1.0)
    return -np.sum(p * np.log(p), axis=1)


def get_metrics_nn(model, X_np, y_true):
    """Evaluate a neural net and return a metrics dict."""
    from src.metrics import compute_metrics

    probs = predict_proba_nn(model, X_np)
    y_pred = probs.argmax(axis=1)
    y_prob = probs[:, 1]

    return compute_metrics(y_true, y_pred, y_prob)