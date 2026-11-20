# HER2 SISH Classifier — Flask web app (SISH-aware feature fusion)

Predicts HER2 status (Amplified / Non-Amplified / Normal) from a SISH slide image.

**Proposed model:** the base paper's 45 colour/texture features are fused with **SISH signal features** —
HER2 (black) and CEP17 (red) dots detected at native resolution, the **HER2/CEP17 ratio**, signal densities and
cluster statistics — plus stain-deconvolution features, and classified by a soft-voting ensemble
(RBF-SVM + LightGBM + Extra Trees). Conformal prediction (α = 0.10) flags uncertain cases for ISH.
A baseline trained on the paper's features only is kept for comparison.

```
her2_flask/
├── app.py                         # Flask server
├── train_models.py                # trains proposed + baseline models
├── sish_features.py               # 86 SISH-aware features (identical to the notebook)
├── her2_models.py                 # model wrapper used when saving/loading
├── data/her2_sish_features.npz    # exported by HER2_SISH_Proposed_Model.ipynb
├── models/her2_sish_models.joblib # created by train_models.py
├── static/  templates/  uploads/
```

## 1. Train on Kaggle (once)
Run `HER2_SISH_Proposed_Model.ipynb` on Kaggle (Run All, CPU is enough) and download `sish_results.zip` from the
Output panel. It contains `figures/` and `tables/` for the report, and `flask_models/` for this website.

## 2. Put the trained models into the website
From `sish_results.zip/flask_models/` copy:

| File | Goes to |
|---|---|
| `her2_sish_models.joblib` | `her2_flask/models/` |
| `her2_sish_features.npz` | `her2_flask/data/` |
| `requirements_web.txt` | `her2_flask/` |

## 3. Install and run (Windows, in this folder)
```bat
venv\Scripts\activate
pip install -r requirements_web.txt
python app.py
```
Open http://127.0.0.1:5000

`requirements_web.txt` pins the exact scikit-learn / LightGBM / NumPy versions the models were trained with.
If pip cannot install those versions for your Python, or the page shows a version warning, retrain the same
models locally instead (5-15 min; `--fast` for 1-3 min):
```bat
pip install -r requirements.txt
python train_models.py
```
`train_models.py` uses `data/her2_sish_features.npz` and the notebook's exact test split.
If you have the dataset images on this computer, `python train_models.py --images "D:\path\to\HER2-SISH"`
extracts the features itself.

## What the page shows
* Prediction of the proposed model, its probability, and the conformal status (confident / confirm with ISH)
* The baseline's prediction for the same image
* **SISH signal analysis**: overlay of detected HER2 (green) and CEP17 (yellow) signals, counts and ratio
* Baseline-vs-proposed metrics on the held-out test set

The automated HER2/CEP17 ratio is a colour-based estimate used as model input; it does not replace manual
signal counting. Research prototype — not for clinical diagnosis.
