"""Train the HER2 website models: proposed SISH-Fusion Soft-Vote vs paper-feature baselines.

Features come from HER2_SISH_Proposed_Model.ipynb (data/her2_sish_features.npz), or are extracted here from a
local copy of the dataset with --images.

    python train_models.py                       # uses data/her2_sish_features.npz  (5-15 min)
    python train_models.py --fast                # smaller search (1-3 min)
    python train_models.py --images D:\\HER2-SISH  # extract features from a local dataset folder first

Output: models/her2_sish_models.joblib
"""
import argparse
import datetime as dt
import os
import time
import warnings
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import sklearn
from scipy.stats import loguniform
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, PowerTransformer, StandardScaler
from sklearn.svm import SVC

import sish_features as F
from her2_models import ColumnModel

warnings.filterwarnings("ignore")
try:
    from lightgbm import LGBMClassifier
except ImportError:
    raise SystemExit("LightGBM is missing. Run:  pip install lightgbm")

BASE = os.path.dirname(os.path.abspath(__file__))
RS = 42
ALPHAS = [0.05, 0.10]
PROPOSED = "SISH-Fusion Soft-Vote"
BASELINE = "Soft-Vote (paper features)"
IMG_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def cols(groups):
    return np.concatenate([np.arange(F.GROUPS[g].start, F.GROUPS[g].stop) for g in groups])


SPACES = {
    "svm": (Pipeline([("scaler", PowerTransformer()),
                      ("clf", SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=RS))]),
            {"scaler": [StandardScaler(), PowerTransformer()], "clf__C": loguniform(1e-1, 1e3),
             "clf__gamma": loguniform(1e-4, 1e0)}),
    "lgbm": (Pipeline([("clf", LGBMClassifier(random_state=RS, n_jobs=1, verbose=-1))]),
             {"clf__n_estimators": [200, 400, 800], "clf__num_leaves": [15, 31, 63],
              "clf__learning_rate": [0.02, 0.05, 0.1], "clf__subsample": [0.7, 0.85, 1.0], "clf__subsample_freq": [1],
              "clf__colsample_bytree": [0.6, 0.8, 1.0], "clf__min_child_samples": [5, 10, 20, 40],
              "clf__reg_lambda": [0, 1, 5]}),
    "et": (Pipeline([("clf", ExtraTreesClassifier(random_state=RS, n_jobs=1))]),
           {"clf__n_estimators": [300, 600, 900], "clf__max_features": ["sqrt", 0.3, 0.5, None],
            "clf__min_samples_leaf": [1, 2, 4], "clf__max_depth": [None, 20, 30]}),
    "rf": (Pipeline([("clf", RandomForestClassifier(random_state=RS, n_jobs=1))]),
           {"clf__n_estimators": [200, 400, 600], "clf__max_depth": [None, 10, 20],
            "clf__max_features": ["sqrt", 0.3, 0.5], "clf__min_samples_leaf": [1, 2, 4]}),
    "gb": (Pipeline([("clf", GradientBoostingClassifier(random_state=RS))]),
           {"clf__n_estimators": [100, 200, 400], "clf__learning_rate": [0.03, 0.05, 0.1],
            "clf__max_depth": [2, 3, 5], "clf__subsample": [0.7, 0.85, 1.0]}),
}
MEMBER_NAMES = {"svm": "RBF-SVM", "lgbm": "LightGBM", "et": "Extra Trees"}


def extract_from_folder(folder, out_path):
    """Label = name of the folder that directly contains each image."""
    from joblib import Parallel, delayed
    files = sorted(p for p in Path(folder).rglob("*") if p.suffix.lower() in IMG_EXT and "mask" not in str(p).lower())
    labels = [p.parent.name for p in files]
    print(f"Extracting features from {len(files)} images: {dict(Counter(labels))}")
    feats = Parallel(n_jobs=-1, verbose=5)(delayed(F.extract_feature_vector)(str(p)) for p in files)
    X = np.array(feats)
    _, keep = np.unique(np.round(X, 8), axis=0, return_index=True)
    keep = np.sort(keep)
    np.savez(out_path, X=X[keep], y=np.array(labels)[keep], g=np.array([str(i) for i in keep]),
             paths=np.array([p.name for p in files])[keep], feature_names=np.array(F.FEATURE_NAMES),
             proposed_groups=np.array(["paper", "stain", "fine_lbp", "signal"]), proposed_k=np.array(0))
    print(f"Saved {out_path} ({len(keep)} unique images)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(BASE, "data", "her2_sish_features.npz"))
    ap.add_argument("--images", help="extract features from this dataset folder first")
    ap.add_argument("--out", default=os.path.join(BASE, "models", "her2_sish_models.joblib"))
    ap.add_argument("--fast", action="store_true")
    args = ap.parse_args()
    n_iter = 10 if args.fast else 25

    if args.images:
        os.makedirs(os.path.dirname(args.data), exist_ok=True)
        extract_from_folder(args.images, args.data)
    if not os.path.exists(args.data):
        raise SystemExit(f"{args.data} not found. Run HER2_SISH_Proposed_Model.ipynb on Kaggle and copy "
                         "her2_sish_features.npz into the data folder, or use --images.")

    d = np.load(args.data, allow_pickle=True)
    X = d["X"].astype(float)
    assert X.shape[1] == F.N_FEATURES, f"expected {F.N_FEATURES} features, got {X.shape[1]}"
    enc = LabelEncoder()
    y = enc.fit_transform(d["y"])
    classes = list(enc.classes_)
    prop_groups = [str(g) for g in d["proposed_groups"]] if "proposed_groups" in d else ["paper", "stain", "fine_lbp", "signal"]
    prop_k = int(d["proposed_k"]) if "proposed_k" in d else 0
    prop_cols, paper_cols = cols(prop_groups), cols(["paper"])
    print(f"{len(y)} images, classes {dict(zip(classes, np.bincount(y)))}")
    print(f"Proposed feature groups: {prop_groups} ({len(prop_cols)} features), MI selection k = {prop_k or 'none'}")

    # Same locked test set as the notebook when available
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RS)
    if "test_idx" in d:
        te = np.asarray(d["test_idx"]); tr = np.setdiff1d(np.arange(len(y)), te)
    else:
        tr, te = next(cv.split(X, y))
    X_tr, X_te, y_tr, y_te = X[tr], X[te], y[tr], y[te]
    print(f"Train {len(tr)} | test {len(te)}\n")

    def tune(key, c):
        pipe, space = SPACES[key]
        rs = RandomizedSearchCV(pipe, space, n_iter=n_iter, cv=cv, scoring="accuracy", n_jobs=-1,
                                random_state=RS, refit=False).fit(X_tr[:, c], y_tr)
        print(f"  tuned {key:<5} on {len(c):>2} features: CV {rs.best_score_*100:6.2f}%")
        return clone(pipe).set_params(**rs.best_params_)

    def vote(m):
        return VotingClassifier([(k, m[k]) for k in ["svm", "lgbm", "et"]], voting="soft")

    t0 = time.time()
    m_prop = {k: tune(k, prop_cols) for k in ["svm", "lgbm", "et"]}
    m_paper = {k: tune(k, paper_cols) for k in ["svm", "lgbm", "et", "rf", "gb"]}
    print(f"Tuning done in {time.time()-t0:.0f}s\n")

    specs = {
        PROPOSED: (ColumnModel(vote(m_prop), prop_cols, prop_k or None), "Proposed"),
        **{f"{MEMBER_NAMES[k]} (SISH features)": (ColumnModel(m_prop[k], prop_cols, prop_k or None), "Proposed member")
           for k in ["svm", "lgbm", "et"]},
        BASELINE: (ColumnModel(vote(m_paper), paper_cols), "Baseline"),
        "Random Forest (paper features)": (ColumnModel(m_paper["rf"], paper_cols), "Base paper"),
        "Gradient Boosting (paper features)": (ColumnModel(m_paper["gb"], paper_cols), "Base paper"),
    }

    amp = next((i for i, c in enumerate(classes) if "amp" in c.lower() and "non" not in c.lower()), None)
    non = next((i for i, c in enumerate(classes) if "non" in c.lower()), None)

    models, rows, qhats = {}, [], {}
    for name, (spec, role) in specs.items():
        cv_acc = cross_val_score(spec, X_tr, y_tr, cv=cv, scoring="accuracy", n_jobs=-1).mean()
        oof = cross_val_predict(spec, X_tr, y_tr, cv=cv, method="predict_proba", n_jobs=-1)
        t1 = time.perf_counter(); m = clone(spec).fit(X_tr, y_tr); train_s = time.perf_counter() - t1
        t1 = time.perf_counter(); pr = m.predict_proba(X_te); infer_ms = (time.perf_counter() - t1) / len(te) * 1000
        p = pr.argmax(1)
        s = 1 - oof[np.arange(len(y_tr)), y_tr]; n = len(s)
        q = {a: float(np.quantile(s, min(1, np.ceil((n + 1) * (1 - a)) / n), method="higher")) for a in ALPHAS}
        sets = pr >= 1 - q[0.10]; single = sets.sum(1) == 1
        rng = np.random.default_rng(RS)
        boot = [accuracy_score(y_te[i], p[i]) for i in (rng.integers(0, len(te), len(te)) for _ in range(2000))]
        mask = np.isin(y_te, [amp, non]) if amp is not None and non is not None else None
        try:
            auc = roc_auc_score(y_te, pr, multi_class="ovr", average="macro") if len(classes) > 2 else roc_auc_score(y_te, pr[:, 1])
        except ValueError:
            auc = float("nan")
        rows.append({
            "model": name, "role": role, "main": name == PROPOSED, "n_features": int(prop_k or len(spec.feat_cols)),
            "test_acc": round(accuracy_score(y_te, p) * 100, 2),
            "ci": [round(v * 100, 1) for v in np.percentile(boot, [2.5, 97.5])],
            "precision": round(precision_score(y_te, p, average="macro", zero_division=0) * 100, 2),
            "recall": round(recall_score(y_te, p, average="macro", zero_division=0) * 100, 2),
            "macro_f1": round(f1_score(y_te, p, average="macro") * 100, 2),
            "amp_f1": round(f1_score(y_te[mask], p[mask], labels=[amp, non], average="macro", zero_division=0) * 100, 2)
            if mask is not None and mask.any() else None,
            "auc": round(float(auc), 3), "cv_acc": round(cv_acc * 100, 2),
            "cp_coverage": round(sets[np.arange(len(te)), y_te].mean() * 100, 1),
            "cp_uncertain": round((~single).mean() * 100, 1),
            "cp_confident_acc": round((sets[single].argmax(1) == y_te[single]).mean() * 100, 1) if single.any() else None,
            "train_s": round(train_s, 2), "infer_ms": round(infer_ms, 3),
        })
        models[name], qhats[name] = m, q
        print(f"  {name:<36} test {rows[-1]['test_acc']:6.2f}%  CV {rows[-1]['cv_acc']:6.2f}%  "
              f"F1 {rows[-1]['macro_f1']:6.2f}  flagged {rows[-1]['cp_uncertain']:5.1f}%")

    pp = models[PROPOSED].predict(X_te)
    bundle = {
        "models": models, "main": PROPOSED, "baseline": BASELINE, "qhat": qhats, "classes": classes,
        "metrics": sorted(rows, key=lambda r: (not r["main"], r["role"] != "Baseline", -r["test_acc"])),
        "proposed_groups": prop_groups, "proposed_k": prop_k,
        "report": classification_report(y_te, pp, target_names=classes, digits=4),
        "confusion_matrix": confusion_matrix(y_te, pp).tolist(),
        "n_train": int(len(tr)), "n_test": int(len(te)),
        "trained_at": dt.datetime.now().strftime("%Y-%m-%d %H:%M"), "sklearn_version": sklearn.__version__,
        "versions": {"scikit-learn": sklearn.__version__, "numpy": np.__version__,
                     "lightgbm": __import__("lightgbm").__version__},
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    joblib.dump(bundle, args.out, compress=3)
    print(f"\nProposed model on the test set:\n{bundle['report']}")
    print(f"Saved -> {args.out} ({os.path.getsize(args.out)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
