import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import wasserstein_distance

from eval_utils.MMD import BMMD, cross_correlation_distribution


def load_array(path: str) -> np.ndarray:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if p.suffix != ".npy":
        raise ValueError(f"Only .npy files are supported, got: {p.suffix}")
    data = np.load(p)
    if data.ndim != 3:
        raise ValueError(
            f"Expected array with shape [N, T, D], got shape {data.shape} from {path}"
        )
    return data.astype(np.float32)


def align_shapes(real: np.ndarray, fake: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = min(real.shape[0], fake.shape[0])
    t = min(real.shape[1], fake.shape[1])
    d = min(real.shape[2], fake.shape[2])
    return real[:n, :t, :d], fake[:n, :t, :d]


def bmmd_corr_distance(real: np.ndarray, fake: np.ndarray, kernel: str = "rbf") -> float:
    real_t = torch.tensor(real).float()
    fake_t = torch.tensor(fake).float()
    real_corr = cross_correlation_distribution(real_t).unsqueeze(-1).permute(1, 0, 2)
    fake_corr = cross_correlation_distribution(fake_t).unsqueeze(-1).permute(1, 0, 2)
    return float(BMMD(real_corr, fake_corr, kernel).mean().cpu().item())


def correlation_fro_distance(real: np.ndarray, fake: np.ndarray) -> float:
    real_flat = real.reshape(-1, real.shape[-1])
    fake_flat = fake.reshape(-1, fake.shape[-1])
    real_corr = np.corrcoef(real_flat, rowvar=False)
    fake_corr = np.corrcoef(fake_flat, rowvar=False)
    real_corr = np.nan_to_num(real_corr, nan=0.0, posinf=0.0, neginf=0.0)
    fake_corr = np.nan_to_num(fake_corr, nan=0.0, posinf=0.0, neginf=0.0)
    return float(np.linalg.norm(real_corr - fake_corr, ord="fro"))


def moment_l1_distance(real: np.ndarray, fake: np.ndarray) -> dict[str, float]:
    real_flat = real.reshape(-1, real.shape[-1])
    fake_flat = fake.reshape(-1, fake.shape[-1])

    real_mean, fake_mean = real_flat.mean(axis=0), fake_flat.mean(axis=0)
    real_std, fake_std = real_flat.std(axis=0), fake_flat.std(axis=0)

    return {
        "mean_l1": float(np.mean(np.abs(real_mean - fake_mean))),
        "std_l1": float(np.mean(np.abs(real_std - fake_std))),
    }


def feature_wasserstein(real: np.ndarray, fake: np.ndarray) -> dict[str, float]:
    real_flat = real.reshape(-1, real.shape[-1])
    fake_flat = fake.reshape(-1, fake.shape[-1])
    distances = []
    for i in range(real_flat.shape[1]):
        distances.append(wasserstein_distance(real_flat[:, i], fake_flat[:, i]))
    distances = np.asarray(distances)
    return {
        "wd_mean": float(distances.mean()),
        "wd_std": float(distances.std()),
        "wd_max": float(distances.max()),
    }


def lag_autocorr_distance(real: np.ndarray, fake: np.ndarray, max_lag: int = 5) -> float:
    max_lag = max(1, min(max_lag, real.shape[1] - 1, fake.shape[1] - 1))

    def per_lag_autocorr(x: np.ndarray, lag: int) -> np.ndarray:
        x0 = x[:, :-lag, :].reshape(-1, x.shape[-1])
        x1 = x[:, lag:, :].reshape(-1, x.shape[-1])
        xc0 = x0 - x0.mean(axis=0, keepdims=True)
        xc1 = x1 - x1.mean(axis=0, keepdims=True)
        num = np.mean(xc0 * xc1, axis=0)
        den = np.std(xc0, axis=0) * np.std(xc1, axis=0)
        ac = np.divide(num, den, out=np.zeros_like(num), where=den > 1e-12)
        return ac

    l1s = []
    for lag in range(1, max_lag + 1):
        real_ac = per_lag_autocorr(real, lag)
        fake_ac = per_lag_autocorr(fake, lag)
        l1s.append(np.mean(np.abs(real_ac - fake_ac)))
    return float(np.mean(l1s))


def maybe_unnormalize_fake(fake: np.ndarray, should_unnormalize: bool) -> np.ndarray:
    if should_unnormalize:
        return (fake + 1.0) * 0.5
    return fake


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare real and generated time-series arrays [N, T, D]."
    )
    parser.add_argument("--real", required=True, help="Path to real .npy array")
    parser.add_argument("--fake", required=True, help="Path to generated .npy array")
    parser.add_argument(
        "--unnormalize-fake",
        action="store_true",
        help="Map fake from [-1, 1] to [0, 1] before comparison",
    )
    parser.add_argument("--max-lag", type=int, default=5, help="Max lag for autocorr")
    parser.add_argument(
        "--save-json",
        default=None,
        help="Optional path to save metrics as JSON",
    )
    args = parser.parse_args()

    real = load_array(args.real)
    fake = load_array(args.fake)
    fake = maybe_unnormalize_fake(fake, args.unnormalize_fake)
    real, fake = align_shapes(real, fake)

    metrics = {
        "shape_used": list(real.shape),
        "bmmd_corr_rbf": bmmd_corr_distance(real, fake, kernel="rbf"),
        "corr_fro": correlation_fro_distance(real, fake),
        "autocorr_l1": lag_autocorr_distance(real, fake, max_lag=args.max_lag),
    }
    metrics.update(moment_l1_distance(real, fake))
    metrics.update(feature_wasserstein(real, fake))

    print("=== Real vs Generated Comparison ===")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"{k}: {v:.8f}")
        else:
            print(f"{k}: {v}")

    if args.save_json:
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"Saved metrics to: {args.save_json}")


if __name__ == "__main__":
    main()
