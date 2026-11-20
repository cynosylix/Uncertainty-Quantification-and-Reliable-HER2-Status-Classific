# Uncertainty Quantification and Reliable HER2 Status Classification (SISH)

Machine-learning classification of HER2 status in breast-cancer **SISH** microscope images, with
**conformal prediction** to flag uncertain cases, and a **Flask web app** that shows the prediction, the detected
HER2 / CEP17 signals and which features drove the decision.

Built on: *S. Pintawong et al., "Conformal Prediction for Uncertainty Quantification and Reliable HER2 Status
Classification in Breast Cancer IHC Images", IEEE Access, vol. 13, 2025.*

**Contribution.** The base paper's colour/texture features were designed for IHC staining. In SISH images HER2 appears
as black signal dots and the CEP17 reference as red dots, and amplification is defined by the HER2/CEP17 ratio.
We add **stain-deconvolution** and **SISH signal features** (dot counts, ratio, clusters, nuclear area, measured at
native resolution) and fuse them with the paper's features.

## Results (held-out test set, 107 images)

| Model | Features | Accuracy | 95% CI | Macro F1 | Amp vs Non-Amp F1 | Sent for ISH check |
|---|---|---|---|---|---|---|
| **Soft-Vote (RBF-SVM + LightGBM + Extra Trees), SISH features** | 31 | **90.7%** | 85.0–95.3 | **89.8** | **93.3** | **3.7%** |
| Soft-Vote, base-paper features (baseline) | 45 | 76.6% | 68.2–84.1 | 70.6 | 66.7 | 33.6% |
| Random Forest, base-paper features | 45 | 74.8% | 65.4–82.2 | 69.0 | 65.9 | 33.6% |
| Gradient Boosting, base-paper features | 45 | 72.9% | 64.5–81.3 | 66.6 | 63.5 | 32.7% |
| Decision Tree, base-paper features | 45 | 69.2% | 59.8–77.6 | 64.6 | 64.4 | 52.3% |

Dataset: [HER2-SISH (Kaggle)](https://www.kaggle.com/datasets/zakaurrehmanmmu/her2-sish-dataset) — 537 images
(Normal 300, Non-Amplified 124, Amplified 113); 5 duplicates removed. Full results, figures and a beginner-level
explanation are in **`HER2_SISH_Project_Report.pdf`**.

## Repository structure

```
├── HER2_SISH_Proposed_Model.ipynb     # MAIN notebook: SISH features, all models, evaluation, export (Colab/Kaggle)
├── HER2_SISH_Project_Report.pdf       # project report + teaching guide + code explanation
└── her2_flask/                        # web app
    ├── app.py                         # Flask server
    ├── sish_features.py               # 86-feature extractor + HER2/CEP17 dot detector (same code as the notebook)
    ├── her2_models.py                 # model wrapper needed to load the saved models
    ├── explain.py                     # "why this prediction" feature-contribution charts
    ├── train_models.py                # retrain the models locally from data/her2_sish_features.npz
    ├── models/her2_sish_models.joblib # trained models
    ├── data/her2_sish_features.npz    # extracted features of all images (for retraining and explanations)
    ├── templates/  static/  uploads/
    ├── requirements.txt               # libraries (any recent versions; use with train_models.py)
    └── requirements_web.txt           # library versions matching the Colab-trained model file
```

## Run the web app

Python 3.10 or newer.

```bash
cd her2_flask
python -m venv venv
venv\Scripts\activate            # Windows   (Linux / macOS: source venv/bin/activate)
pip install -r requirements_web.txt
python app.py
```

Open http://127.0.0.1:5000 and upload a SISH image (for example one from the Kaggle dataset).

If the page says the models could not be loaded (different library versions), retrain them locally — it takes
5–15 minutes and uses the same features and test split:

```bash
pip install -r requirements.txt
python train_models.py            # add --fast for a 1–3 minute run
python app.py
```

## Re-run the experiments

Open `HER2_SISH_Proposed_Model.ipynb` in Google Colab (File → Upload notebook) or Kaggle and choose **Run all**
(CPU is enough, about 1–2 hours on free Colab). It downloads the dataset, extracts the features, runs the ablation,
tunes all models with an equal budget, evaluates them on a locked test set (bootstrap CIs, corrected t-test,
McNemar, conformal prediction, permutation importance) and downloads `sish_results.zip` with every figure, table and
the model files for `her2_flask/`.

## Disclaimer

Research prototype — not for clinical diagnosis. The automated HER2/CEP17 ratio is a colour-based estimate used as
model input, not a replacement for manual signal counting; HER2 status must be confirmed by a pathologist.
