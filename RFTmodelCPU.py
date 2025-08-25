# pip install numpy scipy librosa scikit-learn matplotlib soundfile

import cffi
import os
from pathlib import Path
import pickle
import pandas as pd
import numpy as np
from scipy.io import wavfile
import librosa
from tqdm import tqdm

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import GroupKFold

# -----------------------------
# CONFIG
# -----------------------------
MODEL_PATH   = "bee_audio_rf.pkl"                   #name of the model you want to save
RANDOM_STATE = 42
AUDIO_DIR    = Path("Hive1_12_06_2018")                # folder where your .wav files are

FEATURE_CONFIG = {
    "target_sr": 16000,
    "win_s": 5.0,
    "hop_s": 2.5,
    "n_mels": 64,
    "n_mfcc": 13,
    "resample_type": "kaiser_fast",
}


# -----------------------------
# Audio helpers
# -----------------------------
def to_float_mono(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.dtype.kind == "u":  # unsigned (e.g., uint8)
        max_val = np.iinfo(x.dtype).max  # 255
        x = (x.astype(np.float32) - max_val / 2) / (max_val / 2)
    elif x.dtype.kind == "i":  # signed (e.g., int16)
        max_val = np.iinfo(x.dtype).max  # 32767
        x = x.astype(np.float32) / max_val
    else:  # float WAV
        x = x.astype(np.float32)
    if x.ndim == 2:
        x = x.mean(axis=1)
    return x

def extract_features(y: np.ndarray, sr: int, cfg=FEATURE_CONFIG) -> np.ndarray:
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=cfg["n_mels"])
    log_mel = librosa.power_to_db(mel)
    mfcc = librosa.feature.mfcc(S=log_mel, n_mfcc=cfg["n_mfcc"])
    mfcc_d1 = librosa.feature.delta(mfcc)
    mfcc_d2 = librosa.feature.delta(mfcc, order=2)

    sc = librosa.feature.spectral_centroid(y=y, sr=sr)
    bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    ro = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zc = librosa.feature.zero_crossing_rate(y)

    feats = np.hstack([
        mfcc.mean(axis=1), mfcc.std(axis=1),
        mfcc_d1.mean(axis=1), mfcc_d1.std(axis=1),
        mfcc_d2.mean(axis=1), mfcc_d2.std(axis=1),
        sc.mean(axis=1), sc.std(axis=1),
        bw.mean(axis=1), bw.std(axis=1),
        ro.mean(axis=1), ro.std(axis=1),
        zc.mean(axis=1), zc.std(axis=1),
    ]).astype(np.float32)
    return feats

def extract_features_windows(y: np.ndarray, sr: int, win_s: float = 5.0, hop_s: float = 2.5):
    win = max(1, int(win_s * sr))
    hop = max(1, int(hop_s * sr))
    Xw, intervals = [], []
    if len(y) < win:
        return np.empty((0,)), []
    for start in range(0, len(y) - win + 1, hop):
        seg = y[start:start + win]
        feats = extract_features(seg, sr)
        Xw.append(feats)
        intervals.append((start / sr, (start + win) / sr))
    return np.vstack(Xw), intervals

def extract_features_windows_memmap(raw_pcm, sr: int, win_s: float = 5.0, hop_s: float = 2.5, keep_short: bool = True):
    n = int(raw_pcm.shape[0])
    win = max(1, int(win_s * sr))
    hop = max(1, int(hop_s * sr))

    # If stereo -> mono early
    if raw_pcm.ndim == 2:
        raw_pcm = raw_pcm.mean(axis=1)

    # Handle short files: keep them as a single "window"
    if n < win:
        if not keep_short:
            return np.empty((0,)), []
        y = to_float_mono(raw_pcm)
        sr_eff = sr
        if sr > FEATURE_CONFIG["target_sr"]:
            y = librosa.resample(
                y, orig_sr=sr, target_sr=FEATURE_CONFIG["target_sr"],
                res_type=FEATURE_CONFIG["resample_type"]
            )
            sr_eff = FEATURE_CONFIG["target_sr"]
        feats = extract_features(y, sr_eff)
        return feats.reshape(1, -1), [(0.0, n / sr)]

    # Regular sliding windows
    Xw, intervals = [], []
    for start in range(0, n - win + 1, hop):
        seg = raw_pcm[start:start + win]
        y = to_float_mono(seg)
        sr_eff = sr
        if sr > FEATURE_CONFIG["target_sr"]:
            y = librosa.resample(
                y, orig_sr=sr, target_sr=FEATURE_CONFIG["target_sr"],
                res_type=FEATURE_CONFIG["resample_type"]
            )
            sr_eff = FEATURE_CONFIG["target_sr"]

        feats = extract_features(y, sr_eff)
        Xw.append(feats)
        intervals.append((start / sr, (start + win) / sr))

    return np.vstack(Xw), intervals






def build_dataset_from_metadata_windowed(
    metadata_csv: Path,
    win_s: float = 5.0,
    hop_s: float = 2.5,
    trim_db: float = 30.0,
    path_col: str = "file_path",
    label_col: str = "queen_presence",
):
    import pandas as pd, time
    df = pd.read_csv(metadata_csv)
    X, y, groups = [], [], []

    #df = df.head(5)  # limit to n files for a smoke test

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Files"):
        p = Path(row[path_col])
        label = int(row[label_col])
        try:
            sr, raw = wavfile.read(str(p), mmap=True)
            Xw, _ = extract_features_windows_memmap(raw, sr, win_s=win_s, hop_s=hop_s)
            if Xw.size == 0:
                continue
            # Limit the number of windows per file
            MAX_WINS = 800
            if Xw.shape[0] > MAX_WINS:
                Xw = Xw[:MAX_WINS]

            print(f"{p.name}: {Xw.shape[0]} windows")
            X.append(Xw)
            y.append(np.full((Xw.shape[0],), label, dtype=np.int8))
            groups.extend([p.as_posix()] * Xw.shape[0])
        except Exception as e:
            print(f"[SKIP] {p} - {e}")

    if not X:
        raise RuntimeError(f"No usable rows in {metadata_csv}")
    return np.vstack(X), np.concatenate(y), np.array(groups)



def build_balanced_metadata(metadata_csv, audio_dir=None, queen_label_col="queen_presence", samples_per_class=4000):

    #change size of training data set with samples_per_class

    df = pd.read_csv(metadata_csv)

    if "file_path" in df.columns:
        df = df[df["file_path"].apply(lambda p: Path(p).exists())].copy()
    elif "file_name" in df.columns:
        if audio_dir is None:
            raise ValueError("audio_dir must be provided when only 'file_name' is present")
        df["file_path"] = df["file_name"].apply(lambda s: str(Path(audio_dir) / s))
        df = df[df["file_path"].apply(lambda p: Path(p).exists())].copy()
    else:
        raise KeyError(f"{metadata_csv} must contain 'file_path' or 'file_name'")

    class_counts = df[queen_label_col].value_counts().to_dict()
    print(f"[INFO] Class counts before sampling: {class_counts}")
    if len(class_counts) < 2:
        raise RuntimeError("[STOP] Both queen (1) and no_queen (0) must be present.")

    min_count = min(df[queen_label_col].value_counts().min(), samples_per_class)
    balanced_df = (
        df.groupby(queen_label_col, group_keys=False)
          .sample(n=min_count, random_state=RANDOM_STATE)
          .loc[:, ["file_path", queen_label_col]]
          .reset_index(drop=True)
    )
    print(f"[INFO] Class counts after sampling: {balanced_df[queen_label_col].value_counts().to_dict()}")
    return balanced_df





def train_from_metadata_windowed(metadata_csv: Path, win_s=2.5, hop_s=1.25):
    from sklearn.model_selection import StratifiedShuffleSplit

    print(f"\nBuilding windowed dataset from {metadata_csv.resolve()}")
    X, y, groups = build_dataset_from_metadata_windowed(metadata_csv, win_s=win_s, hop_s=hop_s)
    print(f"Dataset size (windows): {X.shape[0]} samples, feature dim: {X.shape[1]}")

    # --- group-aware train/test split (by file) ---
    gdf = pd.DataFrame({"group": groups, "y": y})
    file_labels = gdf.groupby("group")["y"].mean().ge(0.5).astype(int).reset_index()
    counts = file_labels["y"].value_counts().to_dict()
    print("File-level class counts:", counts)

    if file_labels["y"].nunique() < 2:
        raise RuntimeError("[STOP] Only one file-level class present across files.")

    file_groups = file_labels["group"].values
    file_y = file_labels["y"].values
    sss = StratifiedShuffleSplit(n_splits=1, train_size=0.8, random_state=RANDOM_STATE)
    (train_g_idx, test_g_idx) = next(sss.split(file_groups, file_y))
    train_groups = set(file_groups[train_g_idx])
    test_groups  = set(file_groups[test_g_idx])

    train_mask = np.isin(groups, list(train_groups))
    test_mask  = np.isin(groups,  list(test_groups))

    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]
    group_train = groups[train_mask]

    base = RandomForestClassifier(
        n_estimators=300, class_weight="balanced",
        random_state=RANDOM_STATE, n_jobs=-1
    )
    base.fit(X_train, y_train)

    from collections import Counter
    print("Train window labels:", Counter(y_train))
    print("Test  window labels:", Counter(y_test))

    y_pred = base.predict(X_test)
    print("\nBaseline accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred, labels=[0,1], digits=3, zero_division=0))

    # --- group K-fold CV on train groups ---
    unique_groups = np.unique(group_train).size
    n_splits = min(5, unique_groups)
    if n_splits < 2:
        print("[WARN] <2 training groups; skipping GridSearchCV and using the baseline model.")
        tuned_model = base
    else:
        cv = GroupKFold(n_splits=n_splits)
        param_grid = {
            "n_estimators": [300],
            "max_depth": [None, 8, 12, 16],
            "min_samples_split": [2, 4, 8],
            "min_samples_leaf": [1, 2, 4],
        }
        gs = GridSearchCV(
            RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
            param_grid=param_grid,
            cv=cv,
            n_jobs=-1,
        )
        gs.fit(X_train, y_train, groups=group_train)
        print("Best params:", gs.best_params_)
        print("CV best score:", gs.best_score_)
        print("Test accuracy (tuned):", gs.best_estimator_.score(X_test, y_test))
        tuned_model = gs.best_estimator_

    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"model": tuned_model, "config": FEATURE_CONFIG}, f)
    print(f"Saved model to {MODEL_PATH}")

    return tuned_model, {"y_test": y_test, "y_pred": y_pred}


# --------------------------
# Build LABELS_META directly from disk (file name)
# --------------------------

ROOT = AUDIO_DIR.parent  # parent of folders

def infer_label(name: str):
    n = name.lower()
    if "no_queenbee" in n:
        return 0
    if "queenbee" in n and "no_queenbee" not in n:
        return 1
    return None  # ignore files that don't clearly say QueenBee/NO_QueenBee

rows = []
for p in ROOT.rglob("*.wav"):
    lab = infer_label(p.name)
    if lab is not None:
        rows.append({"file_path": str(p.resolve()), "queen_presence": lab})

labels_df = pd.DataFrame(rows).drop_duplicates("file_path")

print("Disk counts:", labels_df["queen_presence"].value_counts().to_dict())

LABELS_META = Path("labels_metadata.csv")
labels_df.to_csv(LABELS_META, index=False)
print(f"Wrote {len(labels_df)} rows to {LABELS_META}")

# --------------------------
# Balance at the FILE level
# --------------------------
balanced_meta = build_balanced_metadata(LABELS_META)   # no audio_dir here
TEMP_META = LABELS_META.with_name("labels_metadata_balanced.csv")
balanced_meta.to_csv(TEMP_META, index=False)
print(f"[INFO] Balanced file list written to {TEMP_META}")

best_model, eval_dict = train_from_metadata_windowed(
    TEMP_META,
    win_s=FEATURE_CONFIG["win_s"],
    hop_s=FEATURE_CONFIG["hop_s"],
)




# -----------------------------
# Predict on new raw WAVs (helper + example)
# -----------------------------
def predict_file_windowed(model, wav_path, win_s=FEATURE_CONFIG["win_s"], hop_s=FEATURE_CONFIG["hop_s"]):
    sr, raw = wavfile.read(str(wav_path))
    if raw.ndim == 2:
        raw = raw.mean(axis=1)
    y_mono = to_float_mono(raw)

    Xw, intervals = [], []
    win = max(1, int(win_s * sr))
    hop = max(1, int(hop_s * sr))
    if len(y_mono) < win:
        raise ValueError("Audio too short for the chosen window size.")

    for start in range(0, len(y_mono) - win + 1, hop):
        seg = y_mono[start:start + win]

        sr_eff = sr
        if sr > FEATURE_CONFIG["target_sr"]:
            seg = librosa.resample(
                seg,
                orig_sr=sr,
                target_sr=FEATURE_CONFIG["target_sr"],
                res_type=FEATURE_CONFIG["resample_type"],
            )
            sr_eff = FEATURE_CONFIG["target_sr"]

        feats = extract_features(seg, sr_eff)
        Xw.append(feats)
        intervals.append((start / sr, (start + win) / sr))

    Xw = np.vstack(Xw)
    pred  = model.predict(Xw)
    proba = model.predict_proba(Xw)[:, 1]
    return {
        "frame_pred": pred,
        "frame_proba": proba,
        "intervals": intervals,
        "maj_vote": int(pred.mean() >= 0.5),
        "mean_prob": float(proba.mean()),
    }


# Example (uncomment to use):
# test_wav = AUDIO_DIR / "Hive1 12_06_2018_QueenBee____00_10_00.wav"
# res = predict_file_windowed(best_model, test_wav, win_s=5.0, hop_s=2.5)
# print("Per-file majority:", res["maj_vote"], "mean P(queen):", res["mean_prob"])

df_meta = pd.read_csv(TEMP_META)  # instead of LABELS_META
print("Balanced files per class:", df_meta["queen_presence"].value_counts().to_dict())


file_counts = df_meta["queen_presence"].value_counts().to_dict()
print("Files per class:", file_counts)


print("AUDIO_DIR:", AUDIO_DIR.resolve())
print("Example path from LABELS_META:", df_meta["file_path"].iloc[0])
p0 = Path(df_meta["file_path"].iloc[0])
print("Exists?", p0.exists())

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(eval_dict["y_test"], eval_dict["y_pred"], labels=[0,1])
print("Confusion matrix:\n", cm)



df = pd.read_csv(LABELS_META)
df["exists"] = df["file_path"].apply(lambda p: Path(p).exists())
print(df.groupby(["queen_presence","exists"]).size())
print("Sample missing no_queen paths:\n", df[(df.queen_presence==0) & (~df.exists)].head(10)["file_path"])
