#!/usr/bin/env python3
"""Cross-dataset domain adaptation baselines on the shared Baseline-vs-Stress task.

Every method gets the same protocol: labelled source dataset, unlabelled target dataset for
adaptation, evaluation on all target windows, fixed epochs (no target labels are used for model
selection), identical encoder. Methods:

  source_only       MLP trained on the source dataset
  coral             + CORAL loss aligning source/target feature covariances (Sun & Saenko 2016)
  mmd               + multi-kernel MMD between source/target representations (Long et al. 2015)
  dann              + gradient-reversal dataset discriminator (Ganin et al. 2016)
  subject_dann      candidate: adversaries for BOTH dataset and subject identity, so the
                    representation is pushed to drop person- and protocol-specific information
  finetune_k5       supervised personalisation: source_only fine-tuned on 5 labelled windows per class
                    from each target subject, tested on that subject's remaining windows

Each method is run with and without label-free per-subject z-scoring of the inputs.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]
META = {"window_id", "subject_id", "dataset", "label", "timestamp_start", "timestamp_end",
        "subject_uid", "original_label", "harmonized_label", "purity", "cardiac_coverage",
        "self_report_stress", "self_report_stress_delta", "self_report_validated", "sam_valence", "sam_arousal"}
EPOCHS, BATCH, LR, ADAPT_WEIGHT = 60, 128, 1e-3, 1.0


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return -ctx.lam * grad, None


class Net(nn.Module):
    def __init__(self, n_in: int, n_subjects: int):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(n_in, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 32), nn.ReLU())
        self.classifier = nn.Linear(32, 2)
        self.dataset_head = nn.Sequential(nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 2))
        self.subject_head = nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, n_subjects))


def coral(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return ((torch.cov(a.T) - torch.cov(b.T)) ** 2).sum() / (4 * a.shape[1] ** 2)


def mmd(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    z = torch.cat([a, b])
    d2 = torch.cdist(z, z) ** 2
    bandwidth = d2.detach().median().clamp_min(1e-6)
    k = sum(torch.exp(-d2 / (bandwidth * s)) for s in (0.25, 0.5, 1, 2, 4))
    n = len(a)
    return k[:n, :n].mean() + k[n:, n:].mean() - 2 * k[:n, n:].mean()


def train(method: str, xs, ys, subj_s, xt, subj_t, n_subjects: int, seed: int) -> Net:
    torch.manual_seed(seed)
    net = Net(xs.shape[1], n_subjects)
    opt = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=1e-4)
    ce = nn.CrossEntropyLoss()
    class_weight = torch.tensor([1.0 / max((ys == c).float().mean().item(), 1e-6) for c in (0, 1)])
    cls_loss = nn.CrossEntropyLoss(weight=class_weight / class_weight.sum())
    steps = EPOCHS * max(1, len(xs) // BATCH)
    rng = np.random.default_rng(seed)
    net.train()
    for step in range(steps):
        i = torch.from_numpy(rng.choice(len(xs), BATCH))
        j = torch.from_numpy(rng.choice(len(xt), BATCH))
        hs, ht = net.encoder(xs[i]), net.encoder(xt[j])
        loss = cls_loss(net.classifier(hs), ys[i])
        lam = 2 / (1 + np.exp(-10 * step / steps)) - 1  # DANN ramp-up schedule
        if method == "coral":
            loss = loss + ADAPT_WEIGHT * coral(hs, ht)
        elif method == "mmd":
            loss = loss + ADAPT_WEIGHT * mmd(hs, ht)
        elif method in ("dann", "subject_dann"):
            h = GradReverse.apply(torch.cat([hs, ht]), lam)
            domain = torch.cat([torch.zeros(BATCH), torch.ones(BATCH)]).long()
            loss = loss + ce(net.dataset_head(h), domain)
            if method == "subject_dann":
                loss = loss + ce(net.subject_head(h), torch.cat([subj_s[i], subj_t[j]]))
        opt.zero_grad()
        loss.backward()
        opt.step()
    return net.eval()


def evaluate(y_true: np.ndarray, proba: np.ndarray) -> dict[str, float]:
    pred = (proba >= 0.5).astype(int)
    return {"balanced_accuracy": balanced_accuracy_score(y_true, pred), "macro_f1": f1_score(y_true, pred, average="macro"),
            "auroc": roc_auc_score(y_true, proba)}


def predict(net: Net, x: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        return torch.softmax(net.classifier(net.encoder(x)), dim=1)[:, 1].numpy()


def finetune_k(net: Net, xt, yt, subj_t, k: int, seed: int) -> dict[str, float]:
    """Per target subject: adapt on k labelled windows per class, test on the rest of that subject."""
    rng = np.random.default_rng(seed)
    truth, proba = [], []
    for subject in subj_t.unique():
        idx = torch.where(subj_t == subject)[0].numpy()
        support = np.concatenate([rng.permutation(idx[yt[idx].numpy() == c])[:k] for c in (0, 1)])
        query = np.setdiff1d(idx, support)
        if len(set(yt[support].tolist())) < 2 or len(query) == 0:
            continue
        personal = copy.deepcopy(net).train()
        opt = torch.optim.Adam(personal.parameters(), lr=LR)
        for _ in range(30):
            opt.zero_grad()
            nn.functional.cross_entropy(personal.classifier(personal.encoder(xt[support])), yt[support]).backward()
            opt.step()
        truth.append(yt[query].numpy())
        proba.append(predict(personal.eval(), xt[query]))
    return evaluate(np.concatenate(truth), np.concatenate(proba))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path,
                        default=REPO_ROOT / "data" / "processed" / "combined" / "harmonized_windows_v2.csv")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs" / "tables" / "jbhi")
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--features", choices=["all", "physiology"], default="all")
    args = parser.parse_args()

    full = pd.read_csv(args.input)
    features = [c for c in full.columns if c not in META]
    # z-score over each subject's WHOLE recording (label-free), before any class filtering
    g = full.groupby("subject_uid")[features]
    zscored = (full[features] - g.transform("mean")) / g.transform("std").replace(0, 1).fillna(1)
    keep = ((full["purity"] >= 0.8) & full["harmonized_label"].isin(["Baseline", "Stress"])
            & full["dataset"].isin(["WESAD", "PhysioNet"]))
    df, zscored = full[keep].reset_index(drop=True), zscored[keep].reset_index(drop=True)
    if args.features == "physiology":
        features = [f for f in features if f.startswith(("hr_", "hrv_", "eda_", "temp_"))]
    subject_codes = torch.from_numpy(pd.factorize(df["subject_uid"])[0])
    y_all = torch.from_numpy((df["harmonized_label"] == "Stress").to_numpy(int))

    rows = []
    for normalisation in ("none", "subject_zscore"):
        x = (zscored if normalisation == "subject_zscore" else df)[features]
        for source, target in (("WESAD", "PhysioNet"), ("PhysioNet", "WESAD")):
            s, t = (df["dataset"] == source).to_numpy(), (df["dataset"] == target).to_numpy()
            x = x.fillna(x[s].median())  # motion-corrupted cardiac windows: impute with the source median
            mean, std = x[s].mean(), x[s].std().replace(0, 1)  # scaler fitted on the source only
            xs = torch.tensor(((x[s] - mean) / std).to_numpy(), dtype=torch.float32)
            xt = torch.tensor(((x[t] - mean) / std).to_numpy(), dtype=torch.float32).clamp(-10, 10)
            for seed in range(args.seeds):
                for method in ("source_only", "coral", "mmd", "dann", "subject_dann"):
                    net = train(method, xs, y_all[s], subject_codes[s], xt, subject_codes[t], int(subject_codes.max()) + 1, seed)
                    rows.append({"train": source, "test": target, "normalisation": normalisation, "method": method,
                                 "seed": seed, **evaluate(y_all[t].numpy(), predict(net, xt))})
                    if method == "source_only":
                        rows.append({"train": source, "test": target, "normalisation": normalisation,
                                     "method": "finetune_k5 (uses target labels)", "seed": seed,
                                     **finetune_k(net, xt, y_all[t], subject_codes[t], 5, seed)})
            print(f"done {source}->{target} [{normalisation}]", flush=True)

    raw = pd.DataFrame(rows)
    summary = raw.groupby(["train", "test", "normalisation", "method"], sort=False)[
        ["balanced_accuracy", "macro_f1", "auroc"]].agg(["mean", "std"]).round(4)
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw.to_csv(args.output_dir / f"domain_adaptation_{args.features}_raw.csv", index=False)
    summary.reset_index().to_csv(args.output_dir / f"domain_adaptation_{args.features}.csv", index=False)
    print(summary.to_string())


if __name__ == "__main__":
    main()
