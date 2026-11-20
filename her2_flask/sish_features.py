"""SISH-aware handcrafted features for HER2 status classification.

Feature vector (86 values), in this order:
  [  0: 45]  Paper features      - Color (6) + LBP P24/R3 (26) + Haralick (13) on a 128x128 image
  [ 45: 60]  Stain features      - H / E / DAB deconvolution channels: mean, std, p10, p50, p90
  [ 60: 70]  Fine LBP            - LBP P8/R1 histogram (10)
  [ 70: 86]  SISH signal features - HER2 (black/brown) and CEP17 (red) signal dots detected at
                                    native resolution, their ratio, densities and cluster statistics

The first 70 values are identical to her2_features_v3.npz from the benchmark notebook.
This file is used unchanged by the training notebook, train_models.py and the Flask app.
"""
import numpy as np
import mahotas
from PIL import Image
from scipy import ndimage as ndi
from skimage.color import rgb2hed, rgb2lab
from skimage.feature import local_binary_pattern

Image.MAX_IMAGE_PIXELS = None

IMAGE_SIZE = (128, 128)          # paper / stain / fine-LBP features (as in the earlier notebooks)
MAX_PIXELS = 4_000_000           # signal detection runs at native resolution, capped for speed

# ---- signal-detection parameters (tuned on HER2-SISH 40x images; colours on a 0-255 scale) ----
BLACK_MAX_GRAY = 130             # HER2 signals are dark ...
BLACK_MAX_ASTAR = 18             # ... but not magenta (that is CEP17)
BLACK_MAX_BLUE_EXCESS = 25       # ... and not dark-blue nuclear chromatin (B - R)
RED_MIN_ASTAR = 28               # CEP17 signals are strongly red/magenta in Lab a*
RED_MAX_L = 72
NUC_MIN_BLUE_EXCESS = 12         # haematoxylin-stained nuclei are bluish (B - R) ...
NUC_MAX_GRAY = 215               # ... and darker than background
MIN_DOT_AREA = 6                 # px at reference scale (removes noise)
MAX_DOT_AREA = 2500              # px at reference scale (bigger = artefact / fold)
CLUSTER_FACTOR = 3.0             # blob > 3 x typical single-dot area = cluster of signals
SINGLE_DOT_AREA = 40.0           # typical single signal area at reference scale (fallback)

PAPER_NAMES = ([f"color_{s}_{c}" for s in ("mean", "std") for c in "RGB"]
               + [f"lbp24_{i}" for i in range(26)]
               + ["har_" + n for n in ["ASM", "contrast", "correlation", "variance", "IDM", "sum_avg",
                                       "sum_var", "sum_entropy", "entropy", "diff_var", "diff_entropy",
                                       "IMC1", "IMC2"]])
STAIN_NAMES = [f"stain_{ch}_{s}" for ch in ("H", "E", "DAB") for s in ("mean", "std", "p10", "p50", "p90")]
FINE_LBP_NAMES = [f"lbp8_{i}" for i in range(10)]
SIGNAL_NAMES = ["her2_signals", "cep17_signals", "her2_blobs", "cep17_blobs",
                "her2_cep17_ratio", "log2_ratio",
                "her2_per_mpx", "cep17_per_mpx",
                "her2_per_nuclear_area", "cep17_per_nuclear_area",
                "her2_area_frac", "cep17_area_frac", "nuclear_area_frac",
                "her2_mean_blob_area", "her2_cluster_frac", "cep17_mean_blob_area"]
FEATURE_NAMES = PAPER_NAMES + STAIN_NAMES + FINE_LBP_NAMES + SIGNAL_NAMES
GROUPS = {"paper": slice(0, 45), "stain": slice(45, 60), "fine_lbp": slice(60, 70), "signal": slice(70, 86)}
N_FEATURES = len(FEATURE_NAMES)    # 86


# ------------------------------------------------------------------ paper features (unchanged)
def extract_color_features(img):
    return np.concatenate([np.mean(img, axis=(0, 1)), np.std(img, axis=(0, 1))])


def extract_lbp_features(gray):
    P, R = 24, 3
    lbp = local_binary_pattern(gray, P, R, method="uniform")
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, P + 3), density=True)
    return hist


def extract_haralick_features(gray):
    return mahotas.features.haralick((gray // 8).astype(np.uint8), return_mean=True)


# ------------------------------------------------------------------ stain + fine texture
def extract_stain_features(rgb):
    hed = rgb2hed(rgb)
    out = []
    for ch in range(3):
        c = hed[..., ch].ravel()
        out += [c.mean(), c.std(), *np.percentile(c, [10, 50, 90])]
    return np.array(out)


def extract_lbp_fine(gray):
    lbp = local_binary_pattern(gray, 8, 1, method="uniform")
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 11), density=True)
    return hist


# ------------------------------------------------------------------ SISH signal detection
def _load_native(image_source):
    img = Image.open(image_source).convert("RGB")
    w, h = img.size
    scale = 1.0
    if w * h > MAX_PIXELS:
        scale = (MAX_PIXELS / (w * h)) ** 0.5
        img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.BILINEAR)
    return np.asarray(img), scale


def _blobs(mask, min_area, max_area):
    mask = ndi.binary_opening(mask, structure=np.ones((2, 2)))
    lab, n = ndi.label(mask)
    if n == 0:
        return lab, np.array([], dtype=float)
    areas = ndi.sum(mask, lab, index=np.arange(1, n + 1)).astype(float)
    keep = (areas >= min_area) & (areas <= max_area)
    remap = np.zeros(n + 1, dtype=np.int32)
    remap[1:][keep] = np.arange(1, keep.sum() + 1)
    return remap[lab], areas[keep]


def detect_signals(rgb, scale=1.0):
    """Return HER2 / CEP17 / nuclei masks and blob areas (areas converted to reference-scale px)."""
    a = rgb.astype(np.float32)
    gray = a @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    lab = rgb2lab(rgb)
    astar, L = lab[..., 1], lab[..., 0]
    blue_excess = a[..., 2] - a[..., 0]

    s2 = scale ** 2
    min_a, max_a = max(1.0, MIN_DOT_AREA * s2), MAX_DOT_AREA * s2

    her2_mask = (gray < BLACK_MAX_GRAY) & (astar < BLACK_MAX_ASTAR) & (blue_excess < BLACK_MAX_BLUE_EXCESS)
    cep_mask = (astar > RED_MIN_ASTAR) & (L < RED_MAX_L)
    nuc_mask = (blue_excess > NUC_MIN_BLUE_EXCESS) & (gray < NUC_MAX_GRAY)
    nuc_mask = ndi.binary_opening(nuc_mask, structure=np.ones((3, 3)))

    her2_lab, her2_areas = _blobs(her2_mask, min_a, max_a)
    cep_lab, cep_areas = _blobs(cep_mask, min_a, max_a)
    return {"her2_lab": her2_lab, "cep_lab": cep_lab, "nuc_mask": nuc_mask,
            "her2_areas": her2_areas / s2, "cep_areas": cep_areas / s2, "scale": scale}


def _estimate_signals(areas):
    """Cluster-aware signal count: a large blob counts as several signals (area / single-dot area)."""
    if len(areas) == 0:
        return 0.0, 0.0
    single = np.median(areas[areas <= np.percentile(areas, 50)]) if len(areas) >= 5 else SINGLE_DOT_AREA
    single = max(single, MIN_DOT_AREA)
    cluster = areas > CLUSTER_FACTOR * single
    count = (~cluster).sum() + np.round(areas[cluster] / single).sum()
    cluster_frac = areas[cluster].sum() / areas.sum()
    return float(count), float(cluster_frac)


def signal_measurements(rgb, scale=1.0):
    d = detect_signals(rgb, scale)
    n_px_ref = rgb.shape[0] * rgb.shape[1] / scale ** 2
    her2_n, her2_cluster = _estimate_signals(d["her2_areas"])
    cep_n = float(len(d["cep_areas"]))          # CEP17 signals are single dots: one per blob
    nuc_area_ref = d["nuc_mask"].sum() / scale ** 2
    ratio = (her2_n + 1.0) / (cep_n + 1.0)
    m = {
        "her2_signals": her2_n, "cep17_signals": cep_n,
        "her2_blobs": float(len(d["her2_areas"])), "cep17_blobs": float(len(d["cep_areas"])),
        "her2_cep17_ratio": ratio, "log2_ratio": float(np.log2(ratio)),
        "her2_per_mpx": her2_n / n_px_ref * 1e6, "cep17_per_mpx": cep_n / n_px_ref * 1e6,
        "her2_per_nuclear_area": her2_n / (nuc_area_ref + 1) * 1e4,
        "cep17_per_nuclear_area": cep_n / (nuc_area_ref + 1) * 1e4,
        "her2_area_frac": d["her2_areas"].sum() / n_px_ref,
        "cep17_area_frac": d["cep_areas"].sum() / n_px_ref,
        "nuclear_area_frac": nuc_area_ref / n_px_ref,
        "her2_mean_blob_area": float(d["her2_areas"].mean()) if len(d["her2_areas"]) else 0.0,
        "her2_cluster_frac": her2_cluster,
        "cep17_mean_blob_area": float(d["cep_areas"].mean()) if len(d["cep_areas"]) else 0.0,
    }
    return m, d


# ------------------------------------------------------------------ public API
def extract_feature_vector(image_source):
    """86-value SISH-aware feature vector for one image (path or file-like)."""
    native, scale = _load_native(image_source)
    small = Image.fromarray(native).resize(IMAGE_SIZE)
    rgb = np.asarray(small)
    gray = np.asarray(small.convert("L"))
    m, _ = signal_measurements(native, scale)
    v = np.concatenate([
        extract_color_features(rgb), extract_lbp_features(gray), extract_haralick_features(gray),
        extract_stain_features(rgb), extract_lbp_fine(gray),
        np.array([m[k] for k in SIGNAL_NAMES], dtype=float),
    ])
    return np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)


def extract_all_features(image_source):
    """Paper's 45 features only (baseline)."""
    return extract_feature_vector(image_source)[:45]


def _overlay(native, d, max_side):
    out = native.copy()
    for lab, colour in ((d["her2_lab"], (0, 200, 0)), (d["cep_lab"], (255, 200, 0))):
        mask = lab > 0
        edge = mask & ~ndi.binary_erosion(mask, iterations=1)
        edge = ndi.binary_dilation(edge, iterations=1)
        out[edge] = colour
    img = Image.fromarray(out)
    img.thumbnail((max_side, max_side))
    return img


def analyse_image(image_source, max_side=900):
    """One pass for the website: (86-feature vector, overlay PIL image, signal measurements)."""
    native, scale = _load_native(image_source)
    small = Image.fromarray(native).resize(IMAGE_SIZE)
    rgb = np.asarray(small)
    gray = np.asarray(small.convert("L"))
    m, d = signal_measurements(native, scale)
    v = np.concatenate([
        extract_color_features(rgb), extract_lbp_features(gray), extract_haralick_features(gray),
        extract_stain_features(rgb), extract_lbp_fine(gray),
        np.array([m[k] for k in SIGNAL_NAMES], dtype=float),
    ])
    return np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0), _overlay(native, d, max_side), m


def signal_overlay(image_source, max_side=900):
    """RGB overlay for display: HER2 signals outlined green, CEP17 signals outlined yellow.
    Returns (PIL.Image, measurements dict)."""
    native, scale = _load_native(image_source)
    m, d = signal_measurements(native, scale)
    return _overlay(native, d, max_side), m
