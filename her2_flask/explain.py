"""Per-image explanation: which features support the model's prediction.

Method (model-agnostic, replacement-based local importance):
  for each feature the model uses, replace this image's value with the values of N typical training
  images (one at a time, all other features unchanged) and measure how much the probability of the
  predicted class changes on average.
    contribution = P(class | real value) - mean P(class | typical values)       (percentage points)
  > 0  the feature's value SUPPORTS the prediction
  < 0  the feature's value ARGUES AGAINST it
The same is done for whole feature groups (all columns of a group replaced together).
It shows what the model relies on for this image - an association learned from data, not a biological cause.
"""
import numpy as np

from sish_features import FEATURE_NAMES, GROUPS

GROUP_LABEL = {"paper": "Paper colour / texture", "stain": "Stain deconvolution",
               "fine_lbp": "Fine texture (LBP)", "signal": "SISH signals (HER2 / CEP17)"}

_SIGNAL = {
    "her2_signals": "HER2 signal count", "cep17_signals": "CEP17 signal count",
    "her2_blobs": "HER2 blob count", "cep17_blobs": "CEP17 blob count",
    "her2_cep17_ratio": "HER2/CEP17 ratio", "log2_ratio": "log2 HER2/CEP17 ratio",
    "her2_per_mpx": "HER2 per megapixel", "cep17_per_mpx": "CEP17 per megapixel",
    "her2_per_nuclear_area": "HER2 per nuclear area", "cep17_per_nuclear_area": "CEP17 per nuclear area",
    "her2_area_frac": "HER2 area fraction", "cep17_area_frac": "CEP17 area fraction",
    "nuclear_area_frac": "Nuclear area fraction", "her2_mean_blob_area": "Mean HER2 blob size",
    "her2_cluster_frac": "HER2 area in clusters", "cep17_mean_blob_area": "Mean CEP17 blob size",
}
_STAT = {"mean": "mean", "std": "spread", "p10": "10th pct", "p50": "median", "p90": "90th pct"}
_CH = {"H": "Haematoxylin", "E": "Eosin", "DAB": "DAB / silver"}
_HAR = {"ASM": "energy", "IDM": "homogeneity", "sum_avg": "sum average", "sum_var": "sum variance",
        "sum_entropy": "sum entropy", "diff_var": "difference variance", "diff_entropy": "difference entropy"}


def friendly(name):
    if name in _SIGNAL:
        return _SIGNAL[name]
    p = name.split("_")
    if p[0] == "stain":
        return f"{_CH[p[1]]} {_STAT[p[2]]}"
    if p[0] == "color":
        return f"{ {'R': 'Red', 'G': 'Green', 'B': 'Blue'}[p[2]] } colour {'mean' if p[1] == 'mean' else 'spread'}"
    if p[0] == "lbp24":
        return f"Texture LBP-24 bin {p[1]}"
    if p[0] == "lbp8":
        return f"Fine texture LBP-8 bin {p[1]}"
    if p[0] == "har":
        key = name[4:]
        return f"Haralick {_HAR.get(key, key)}"
    return name


def group_of(j):
    for g, s in GROUPS.items():
        if s.start <= j < s.stop:
            return g
    return "other"


def fmt(v):
    v = float(v)
    if v == int(v) and abs(v) < 1e6:
        return f"{int(v):,}"
    if abs(v) >= 100:
        return f"{v:,.0f}"
    if abs(v) >= 1:
        return f"{v:.2f}"
    return f"{v:.3g}"


class Explainer:
    def __init__(self, npz_path, classes, n_background=30, seed=0):
        d = np.load(npz_path, allow_pickle=True)
        X = d["X"].astype(float)
        y = np.asarray(d["y"]).astype(str)
        idx = np.arange(len(X))
        if "test_idx" in d.files:                       # use training images only
            idx = np.setdiff1d(idx, np.asarray(d["test_idx"]))
        Xtr, ytr = X[idx], y[idx]
        rng = np.random.default_rng(seed)
        self.background = Xtr[rng.choice(len(Xtr), size=min(n_background, len(Xtr)), replace=False)]
        self.typical = {c: np.median(Xtr[ytr == c], axis=0) for c in classes if (ytr == c).any()}

    def explain(self, model, x, cls_idx, cols, top=10):
        x = np.asarray(x, dtype=float).ravel()
        cols = [int(j) for j in cols]
        B = len(self.background)
        groups = {}
        for j in cols:
            groups.setdefault(group_of(j), []).append(j)
        units = [[j] for j in cols] + list(groups.values())

        rows = np.repeat(x[None, :], len(units) * B, axis=0)
        for u, cs in enumerate(units):
            rows[u * B:(u + 1) * B, cs] = self.background[:, cs]
        base = float(model.predict_proba(x[None, :])[0][cls_idx])
        p = model.predict_proba(rows)[:, cls_idx].reshape(len(units), B).mean(1)
        contrib = (base - p) * 100

        feats = [{"j": j, "name": friendly(FEATURE_NAMES[j]), "group": GROUP_LABEL.get(group_of(j), ""),
                  "value": fmt(x[j]), "contrib": float(c),
                  "typical": {k: fmt(v[j]) for k, v in self.typical.items()}}
                 for j, c in zip(cols, contrib[:len(cols)])]
        feats.sort(key=lambda f: -abs(f["contrib"]))
        grp = [{"name": GROUP_LABEL.get(g, g), "contrib": float(c), "n": len(cs)}
               for (g, cs), c in zip(groups.items(), contrib[len(cols):])]
        grp.sort(key=lambda g: -abs(g["contrib"]))
        return base * 100, feats[:top], grp


def chart(items, label_w=236, mid=500, half=180, row_h=30, top=16):
    """Geometry for a diverging horizontal bar chart (SVG drawn in the template)."""
    m = max([abs(i["contrib"]) for i in items] + [1e-9])
    out = []
    for k, it in enumerate(items):
        w = abs(it["contrib"]) / m * half
        pos = it["contrib"] >= 0
        out.append({**it, "y": top + k * row_h, "x": mid if pos else mid - w, "w": max(w, 1.5), "pos": pos,
                    "tx": mid + w + 6 if pos else mid - w - 6, "anchor": "start" if pos else "end",
                    "label": f"{it['contrib']:+.1f} pp"})
    return {"rows": out, "height": top + len(items) * row_h + 10, "mid": mid, "label_x": label_w}
