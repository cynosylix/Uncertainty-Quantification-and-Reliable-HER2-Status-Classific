"""HER2 Status Classifier - Flask web app (SISH-aware feature fusion).

First time:
    pip install -r requirements.txt
    python train_models.py          # builds models/her2_sish_models.joblib
Then:
    python app.py                   # open http://127.0.0.1:5000
"""
import os
import uuid

import joblib
import numpy as np
from flask import Flask, flash, redirect, render_template, request, send_from_directory, url_for
from werkzeug.utils import secure_filename

import sish_features as F
import her2_models  # noqa: F401  (needed to load the saved ColumnModel objects)
from explain import Explainer, chart

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "her2_sish_models.joblib")
FEATURES_PATH = os.path.join(BASE_DIR, "data", "her2_sish_features.npz")   # background images for explanations
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

ALLOWED_EXT = {"png", "jpg", "jpeg", "tif", "tiff", "bmp"}
ALPHA = 0.10                 # conformal significance level -> 90 % target coverage
CLINICAL_RATIO = 2.0         # ASCO/CAP: HER2/CEP17 ratio >= 2.0 = amplified

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "her2-dev-key")
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024


def display_info(cls):
    c = cls.lower()
    if "non" in c:
        return {"name": "HER2 Non-Amplified", "note": "HER2-negative pattern", "key": "neg"}
    if "amp" in c:
        return {"name": "HER2 Amplified", "note": "HER2-positive pattern - candidate for targeted therapy", "key": "pos"}
    if "normal" in c:
        return {"name": "Normal tissue", "note": "No tumour region detected", "key": "normal"}
    return {"name": cls, "note": "", "key": "neg"}


BUNDLE, LOAD_ERROR = None, None
if os.path.exists(MODEL_PATH):
    try:
        BUNDLE = joblib.load(MODEL_PATH)
        print(f"Loaded {len(BUNDLE['models'])} models (trained {BUNDLE['trained_at']}). Main: {BUNDLE['main']}")
    except Exception as e:
        LOAD_ERROR = f"Could not load the models ({e}). Run  python train_models.py  again."
else:
    LOAD_ERROR = "No trained models yet. Run  python train_models.py  once, then restart the app."
if LOAD_ERROR:
    print("WARNING:", LOAD_ERROR)

VERSION_WARNING = None
if BUNDLE and BUNDLE.get("versions"):
    import sklearn, lightgbm
    have = {"scikit-learn": sklearn.__version__, "lightgbm": lightgbm.__version__, "numpy": np.__version__}
    diff = [f"{k} {BUNDLE['versions'][k]} (installed {v})" for k, v in have.items()
            if k in BUNDLE["versions"] and BUNDLE["versions"][k].split(".")[:2] != v.split(".")[:2]]
    if diff:
        VERSION_WARNING = ("The models were trained with " + ", ".join(diff) + ". If predictions fail or look wrong, "
                           "run  pip install -r requirements_web.txt  or retrain here with  python train_models.py.")
        print("WARNING:", VERSION_WARNING)

CLASSES = BUNDLE["classes"] if BUNDLE else []
CLASS_INFO = {c: display_info(c) for c in CLASSES}

EXPLAINER = None
if BUNDLE and os.path.exists(FEATURES_PATH):
    try:
        EXPLAINER = Explainer(FEATURES_PATH, CLASSES)
    except Exception as e:
        print("WARNING: explanations disabled -", e)
elif BUNDLE:
    print("WARNING: explanations disabled - data/her2_sish_features.npz not found")


def explain_prediction(vec, label):
    model = BUNDLE["models"][BUNDLE["main"]]
    cols = getattr(model, "feat_cols", range(F.N_FEATURES))
    base, feats, groups = EXPLAINER.explain(model, vec, CLASSES.index(label), cols)
    return {"cls": CLASS_INFO[label]["name"], "key": CLASS_INFO[label]["key"], "prob": round(base, 1),
            "features": chart(feats), "groups": chart(groups),
            "typical_names": [CLASS_INFO[c]["name"] for c in CLASSES if c in EXPLAINER.typical],
            "typical_keys": [c for c in CLASSES if c in EXPLAINER.typical]}


def allowed_file(name):
    return "." in name and name.rsplit(".", 1)[1].lower() in ALLOWED_EXT


def predict(image_path, overlay_path):
    vec, overlay, sig = F.analyse_image(image_path)
    overlay.save(overlay_path)
    x = vec.reshape(1, -1)
    main = BUNDLE["main"]

    rows = []
    for name, model in BUNDLE["models"].items():
        proba = model.predict_proba(x)[0]
        idx = int(np.argmax(proba))
        rows.append({"model": name, "is_main": name == main, "is_baseline": name == BUNDLE["baseline"],
                     "label": CLASSES[idx], "display": CLASS_INFO[CLASSES[idx]]["name"],
                     "key": CLASS_INFO[CLASSES[idx]]["key"], "confidence": round(float(proba[idx]) * 100, 2),
                     "probs": [round(float(p) * 100, 2) for p in proba]})
    rows.sort(key=lambda r: (not r["is_main"], not r["is_baseline"], -r["confidence"]))
    m = rows[0]

    in_set = np.array(m["probs"]) / 100 >= 1 - BUNDLE["qhat"][main][ALPHA]
    pred_set = [CLASS_INFO[c]["name"] for c, keep in zip(CLASSES, in_set) if keep]
    base = next(r for r in rows if r["is_baseline"])

    ratio = sig["her2_cep17_ratio"]
    signals = {
        "her2": int(round(sig["her2_signals"])), "cep17": int(round(sig["cep17_signals"])),
        "ratio": round(ratio, 2), "clusters": round(sig["her2_cluster_frac"] * 100, 1),
        "nuclear": round(sig["nuclear_area_frac"] * 100, 1),
        "ratio_says": "amplified range" if ratio >= CLINICAL_RATIO else "non-amplified range",
        "ratio_key": "pos" if ratio >= CLINICAL_RATIO else "neg",
    }
    result = {**m, "info": CLASS_INFO[m["label"]], "pred_set": pred_set,
              "confident": len(pred_set) == 1 and pred_set[0] == m["display"],
              "probs_named": [{"name": CLASS_INFO[c]["name"], "key": CLASS_INFO[c]["key"], "p": p}
                              for c, p in zip(CLASSES, m["probs"])],
              "baseline": base, "agree_baseline": base["label"] == m["label"]}
    explanation = None
    if EXPLAINER:
        try:
            explanation = explain_prediction(vec, m["label"])
        except Exception as e:
            print("WARNING: explanation failed -", e)
    return result, rows, signals, explanation


def page(**kw):
    main_metrics = base_metrics = None
    if BUNDLE:
        main_metrics = next(r for r in BUNDLE["metrics"] if r["model"] == BUNDLE["main"])
        base_metrics = next(r for r in BUNDLE["metrics"] if r["model"] == BUNDLE["baseline"])
    return render_template("index.html", error=LOAD_ERROR, version_warning=VERSION_WARNING, bundle=BUNDLE,
                           mm=main_metrics, bm=base_metrics,
                           alpha=ALPHA, clinical_ratio=CLINICAL_RATIO,
                           class_names=[CLASS_INFO[c]["name"] for c in CLASSES], **kw)


@app.route("/")
def index():
    return page()


@app.route("/predict", methods=["POST"])
def predict_route():
    if LOAD_ERROR:
        flash(LOAD_ERROR)
        return redirect(url_for("index"))
    file = request.files.get("image")
    if not file or file.filename == "":
        flash("Please choose an image to upload.")
        return redirect(url_for("index"))
    if not allowed_file(file.filename):
        flash("Unsupported file type. Use PNG, JPG, TIFF or BMP.")
        return redirect(url_for("index"))

    uid = uuid.uuid4().hex
    path = os.path.join(UPLOAD_DIR, f"{uid}.{file.filename.rsplit('.', 1)[1].lower()}")
    file.save(path)
    overlay_name = f"{uid}_signals.png"
    try:
        result, rows, signals, explanation = predict(path, os.path.join(UPLOAD_DIR, overlay_name))
    except Exception as e:
        flash(f"Could not process this image: {e}")
        return redirect(url_for("index"))
    return page(result=result, rows=rows, signals=signals, explanation=explanation,
                overlay_url=url_for("uploaded_file", filename=overlay_name),
                original_name=secure_filename(file.filename))


@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOAD_DIR, filename)


@app.errorhandler(413)
def too_large(_):
    flash("File is too large (max 50 MB).")
    return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
