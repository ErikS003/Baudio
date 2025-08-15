# minirocket_fft_10pct.py
import ast, re, os, difflib
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from joblib import dump

from sktime.transformations.panel.rocket import MiniRocket

# --------------------
# Config
# --------------------
CSV_PATH = "rms_presence_pairs.csv"               # <-- change this
WANTED_LABEL = "queen_presence"     # <-- your label column
SUBSET_FRAC = 0.10                  # 10%
RANDOM_STATE = 42
KNOWN_LEN = 16000                   # 2 s @ 8 kHz (set None to auto-detect)
SR = 8000                           # sampling rate (Hz)
KEEP_BINS = 1024                    # compress FFT to this many bins
USE_LOG_MAG = True                  # log magnitude (dB-like)
EPS = 1e-10

# --------------------
# Helpers
# --------------------
def find_col(df, wanted):
    df = df.rename(columns={c: c.strip() for c in df.columns})
    cols_lower = {c.lower(): c for c in df.columns}
    if wanted.lower() in cols_lower:
        return df, cols_lower[wanted.lower()]
    match = difflib.get_close_matches(wanted.lower(), cols_lower.keys(), n=1, cutoff=0.8)
    if match:
        return df, cols_lower[match[0]]
    raise KeyError(f"Couldn't find label column {wanted!r}. Available: {list(df.columns)}")

_num_re = re.compile(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?")
def parse_series(cell):
    if isinstance(cell, (list, tuple, np.ndarray)):
        return np.asarray(cell, dtype=np.float32).ravel()
    if pd.isna(cell):
        return np.array([], dtype=np.float32)
    s = str(cell).strip()
    if s.startswith("[") and s.endswith("]"):
        try:
            return np.asarray(ast.literal_eval(s), dtype=np.float32).ravel()
        except Exception:
            pass
    toks = _num_re.findall(s)
    if toks:
        return np.asarray([float(t) for t in toks], dtype=np.float32).ravel()
    cleaned = re.sub(r"[\[\]]", " ", s).replace(",", " ")
    arr = np.fromstring(cleaned, sep=" ", dtype=np.float32)
    return arr.ravel()

def fix_length(x, L):
    if len(x) == L: return x
    if len(x) > L:  return x[:L]
    out = np.zeros(L, dtype=np.float32); out[:len(x)] = x; return out

def rfft_to_features(y, keep_bins=KEEP_BINS, use_log=USE_LOG_MAG, eps=EPS):
    """
    y: 1-D time-domain signal (float32), length ~ KNOWN_LEN
    Returns: 1-D spectrum feature vector of fixed length keep_bins
    """
    n = len(y)
    if n == 0:
        return np.zeros(keep_bins, dtype=np.float32)

    # Hann window to reduce spectral leakage
    w = np.hanning(n).astype(np.float32)
    Y = np.fft.rfft(y * w)                # length = n//2 + 1
    mag = np.abs(Y).astype(np.float32)

    if use_log:
        feat = np.log(mag + eps)          # log-magnitude
    else:
        feat = mag

    # Compress or expand to keep_bins via interpolation
    if feat.shape[0] != keep_bins:
        x_old = np.linspace(0, 1, num=feat.shape[0], dtype=np.float32)
        x_new = np.linspace(0, 1, num=keep_bins, dtype=np.float32)
        feat = np.interp(x_new, x_old, feat).astype(np.float32)

    return feat

def to_nested_univariate(X_2d: np.ndarray) -> pd.DataFrame:
    # sktime wants a DataFrame where each cell is a pd.Series
    return pd.DataFrame({"dim_0": [pd.Series(row) for row in X_2d]})

# --------------------
# Load CSV & subset 10%
# --------------------
df_full = pd.read_csv(CSV_PATH)
df_full, LABEL_COL = find_col(df_full, WANTED_LABEL)
if "rms" not in df_full.columns:
    raise KeyError("CSV must have an 'rms' column containing your time series values.")

y_full = df_full[LABEL_COL].astype(str).to_numpy()
sss = StratifiedShuffleSplit(n_splits=1, test_size=SUBSET_FRAC, random_state=RANDOM_STATE)
_, subset_idx = next(sss.split(df_full, y_full))
df = df_full.iloc[subset_idx].reset_index(drop=True)

print(f"Using {len(df)} rows ({100*SUBSET_FRAC:.1f}% subset) from {df_full[LABEL_COL].nunique()} classes.")

# --------------------
# Parse time series -> FFT features
# --------------------
series_list = [parse_series(x) for x in df["rms"].tolist()]
if KNOWN_LEN is None:
    TARGET_LEN = max(len(x) for x in series_list)
else:
    TARGET_LEN = KNOWN_LEN

# drop empty rows up front
keep_mask = np.array([len(x) > 0 for x in series_list])
dropped = int((~keep_mask).sum())
if dropped:
    print(f"Dropping {dropped} rows with empty 'rms' series.")
series_list = [series_list[i] for i in range(len(series_list)) if keep_mask[i]]
df = df.loc[keep_mask].reset_index(drop=True)

# pad/trim then FFT -> features
X_freq = []
for x in series_list:
    x = fix_length(x, TARGET_LEN)
    # (Optional) If your CSV values are already power-normalized to RMS=1, skip this:
    # normalize to RMS=1 for consistency
    rms = float(np.sqrt(np.mean(x**2))) if x.size else 0.0
    if rms > 0:
        x = x / rms
    spec = rfft_to_features(x, keep_bins=KEEP_BINS, use_log=USE_LOG_MAG, eps=EPS)
    X_freq.append(spec)

X_freq = np.stack(X_freq, axis=0)     # shape: (n_samples, KEEP_BINS)
y = df[LABEL_COL].astype(str).to_numpy()

print(f"Feature matrix: {X_freq.shape}  (~{X_freq.nbytes/1e6:.1f} MB in memory)")

# Convert to sktime nested: treat frequency bins as the "time" axis
X_nested = to_nested_univariate(X_freq)

# --------------------
# Train/test & MiniROCKET
# --------------------
X_tr, X_te, y_tr, y_te = train_test_split(
    X_nested, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

pipe = Pipeline(steps=[
    ("minirocket", MiniRocket(random_state=RANDOM_STATE)),
    ("clf", RidgeClassifierCV(alphas=np.logspace(-3, 3, 7)))
])

pipe.fit(X_tr, y_tr)
y_pred = pipe.predict(X_te)

print("Accuracy:", accuracy_score(y_te, y_pred))
print("\nClassification report:\n", classification_report(y_te, y_pred))

# --------------------
# Save pipeline
# --------------------
os.makedirs("artifacts", exist_ok=True)
dump(pipe, "artifacts/minirocket_fft_10pct.joblib")
print("\nSaved pipeline to artifacts/minirocket_fft_10pct.joblib")
