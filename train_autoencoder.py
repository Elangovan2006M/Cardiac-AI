"""PyTorch Conv-AE training and EVT threshold pipeline."""
import os
import json
import numpy as np
from scipy.stats import genpareto
from sklearn.metrics import roc_auc_score, f1_score

from dataset_generator import generate_full_dataset, DATASET_PATH

np.random.seed(42)
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODEL_DIR, exist_ok=True)


def load_dataset():
    if not os.path.exists(DATASET_PATH):
        return generate_full_dataset(1500)
    with np.load(DATASET_PATH) as data:
        X_normal = data['X_normal']
        X_abnormal = data['X_abnormal']
    print(f"Loaded dataset: {len(X_normal)} normal, {len(X_abnormal)} abnormal samples.")
    return X_normal, X_abnormal


def fit_evt_threshold(normal_mses, percentile=90.0, target_fpr=0.01):
    normal_mses = np.asarray(normal_mses, dtype=np.float64)
    normal_mses = normal_mses[np.isfinite(normal_mses) & (normal_mses >= 0)]
    fallback = float(np.percentile(normal_mses, 99.0)) if len(normal_mses) else 0.0
    u = float(np.percentile(normal_mses, percentile)) if len(normal_mses) else fallback
    exceedances = normal_mses[normal_mses > u] - u

    if len(exceedances) < 10:
        return fallback, u, 0.0, 1.0

    try:
        c, _loc, scale = genpareto.fit(exceedances)
        N = len(normal_mses)
        Nu = len(exceedances)
        if not np.isfinite(c) or not np.isfinite(scale) or scale <= 0:
            raise ValueError("Invalid GPD parameters")
        if abs(c) > 1e-6:
            t_evt = u + (scale / c) * ((((N / Nu) * target_fpr) ** (-c)) - 1.0)
        else:
            t_evt = u - scale * np.log((N / Nu) * target_fpr)
    except Exception:
        return fallback, u, 0.0, 1.0

    if not np.isfinite(t_evt) or t_evt < 0:
        t_evt = fallback
    return float(t_evt), float(u), float(c), float(scale)


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class PyTorchConvAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1), nn.BatchNorm2d(16), nn.ReLU(True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(True),
            nn.Conv2d(32, 16, 1), nn.BatchNorm2d(16), nn.ReLU(True)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(16, 32, 3, stride=2, padding=1, output_padding=1), nn.BatchNorm2d(32), nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1), nn.BatchNorm2d(16), nn.ReLU(True),
            nn.Conv2d(16, 1, 1), nn.Sigmoid()
        )

    def forward(self, x):
        latent = self.encoder(x)
        return self.decoder(latent), latent


def train_model():
    X_normal, X_abnormal = load_dataset()
    train_end = int(0.70 * len(X_normal))
    val_end = int(0.85 * len(X_normal))
    X_train = torch.tensor(X_normal[:train_end]).unsqueeze(1)
    X_val = torch.tensor(X_normal[train_end:val_end]).unsqueeze(1)
    X_test_norm = torch.tensor(X_normal[val_end:]).unsqueeze(1)
    X_test_abnorm = torch.tensor(X_abnormal).unsqueeze(1)

    torch.manual_seed(42)
    model = PyTorchConvAutoencoder()
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    loader = DataLoader(TensorDataset(X_train, X_train), batch_size=32, shuffle=True)

    model.train()
    for epoch in range(40):
        total_loss = 0.0
        for batch_x, _ in loader:
            optimizer.zero_grad()
            recon, _ = model(batch_x)
            loss = criterion(recon, batch_x)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch_x.size(0)
        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1}/40] Loss: {total_loss / len(X_train):.6f}")

    model.eval()
    with torch.no_grad():
        val_recon, _ = model(X_val)
        val_mses = torch.mean((X_val - val_recon) ** 2, dim=[1, 2, 3]).numpy()
        norm_recon, _ = model(X_test_norm)
        norm_mses = torch.mean((X_test_norm - norm_recon) ** 2, dim=[1, 2, 3]).numpy()
        abnorm_recon, _ = model(X_test_abnorm)
        abnorm_mses = torch.mean((X_test_abnorm - abnorm_recon) ** 2, dim=[1, 2, 3]).numpy()

    t_evt, u, c, scale = fit_evt_threshold(val_mses)
    y_true = np.concatenate([np.zeros_like(norm_mses), np.ones_like(abnorm_mses)])
    y_scores = np.concatenate([norm_mses, abnorm_mses])
    auc = float(roc_auc_score(y_true, y_scores))
    f1 = float(f1_score(y_true, (y_scores >= t_evt).astype(int)))

    model_path = os.path.join(MODEL_DIR, "conv_autoencoder.pt")
    tmp_model_path = model_path + ".tmp"
    torch.save(model.state_dict(), tmp_model_path)
    os.replace(tmp_model_path, model_path)

    meta = {
        "framework": "pytorch", "dataset_size": 1500,
        "evt_threshold": float(t_evt), "u": float(u), "gpd_c": float(c), "gpd_scale": float(scale),
        "roc_auc": auc, "f1_score": f1,
        "normal_mse_mean": float(np.mean(val_mses)), "abnormal_mse_mean": float(np.mean(abnorm_mses)),
        "input_shape": [32, 32]
    }
    meta_path = os.path.join(MODEL_DIR, "evt_metadata.json")
    tmp_meta_path = meta_path + ".tmp"
    with open(tmp_meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    os.replace(tmp_meta_path, meta_path)
    print(f"Saved model to {model_path}")
    return True


train_pytorch = train_model

if __name__ == "__main__":
    train_model()
