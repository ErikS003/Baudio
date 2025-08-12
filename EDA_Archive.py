import re
from pathlib import Path
import numpy as np
import pandas as pd
import librosa

# --- Count labels robustly ---
vc = df_concat["queen presence"].value_counts(dropna=False)
Q_1 = int(vc.get(1, 0))
Q_0 = int(vc.get(0, 0))
Q_nan = int(vc.get(np.nan, 0))
print(f"queen present: {Q_1}\nqueen not present: {Q_0}\nnot recorded: {Q_nan}\nTotal: {len(df_concat)}")

# --- Keep only the two columns we need and drop rows missing either one ---
df_queen = df_concat[["queen presence", "file name"]].dropna(subset=["queen presence", "file name"]).copy()

# --- 1) Build a recursive file map that is case-insensitive for the extension ---
path_to_dataset = Path("./data/archive/")
wav_map_raw = {}
for p in path_to_dataset.rglob("*"):
    try:
        if p.is_file() and p.suffix.lower() == ".wav":
            wav_map_raw[p.name] = p
    except OSError:
        pass  # ignore any odd filesystem entries

# --- 2) Normalization function applied to BOTH df and disk filenames ---
def norm_key(name: str) -> str:
    # keep only the base name, lower-case
    base = str(name).strip().replace("\\", "/").split("/")[-1].lower()
    # drop any leading numeric prefix like "12_" or "0003-"
    base = re.sub(r"^\d+[_\-]+", "", base)
    # collapse multiple spaces/underscores
    base = re.sub(r"\s+", " ", base).strip()
    base = re.sub(r"_+", "_", base)
    return base

wav_map = {norm_key(k): v for k, v in wav_map_raw.items()}

# --- 3) Normalize DF filenames and map to actual paths ---
df_queen["key"] = df_queen["file name"].astype(str).map(norm_key)
df_queen["wav_path"] = df_queen["key"].map(wav_map)

missing = df_queen["wav_path"].isna().sum()
print(f"Missing files after normalization: {missing} / {len(df_queen)}")
if missing:
    print("Examples of missing names:",
          df_queen.loc[df_queen["wav_path"].isna(), "file name"].unique()[:5])

# --- 4) ONLY proceed with rows that resolved to a real path ---
df_use = df_queen.dropna(subset=["wav_path"]).copy()

# --- 5) Feature extraction helpers (robust to bad files) ---
def extract_single(p: Path, sr=22050, n_fft=2048, hop_len=512):
    # load audio; if it fails, return None so we can skip
    try:
        y, _ = librosa.load(str(p), sr=sr, mono=True)
    except Exception as e:
        print(f"Failed to load {p}: {e}")
        return None
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_len))
    # MFCC expects power or log-mel; using power from |S|^2 converted to dB is fine here
    feats = np.vstack([
        librosa.feature.mfcc(S=librosa.power_to_db(S**2), n_mfcc=13, hop_length=hop_len),
        librosa.feature.spectral_centroid(S=S, hop_length=hop_len),
        librosa.feature.spectral_bandwidth(S=S, hop_length=hop_len),
        librosa.feature.spectral_rolloff(S=S, hop_length=hop_len),
        librosa.feature.zero_crossing_rate(y, frame_length=n_fft, hop_length=hop_len),
        librosa.feature.chroma_stft(S=S, sr=sr, hop_length=hop_len),
    ])
    stats = np.vstack([feats.mean(axis=1), feats.std(axis=1),
                       feats.min(axis=1), feats.max(axis=1)]).T.ravel()
    return stats

# --- 6) Extract features, skipping any unresolved/bad rows ---
rows = []
for _, row in df_use.iterrows():
    vec = extract_single(row["wav_path"])
    if vec is None:
        continue
    rows.append([row["file name"], *vec, int(row["queen presence"])])

if not rows:
    raise RuntimeError("No features extracted. Check filename normalization and paths above.")

cols = ["file_name"] + [f"f{i}" for i in range(len(rows[0]) - 2)] + ["queen_presence"]
df_features = pd.DataFrame(rows, columns=cols)
print(df_features.head())
