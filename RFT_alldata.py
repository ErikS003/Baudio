# ============================================================
# Train on TBON / SBCM / NUHIVE / BAD using .h5 ground-truth
# ============================================================
# Requires: pip install tables   (PyTables for reading HDF5 with pandas)

from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import torch
import librosa
import torch.nn as nn
import torch.optim as optim
from scipy.io import wavfile

from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import GroupKFold, StratifiedShuffleSplit
N = 0
FEATURE_CONFIG = {
    "target_sr": 16000,
    "win_s": 5.0,
    "hop_s": 2.5,
    "n_mels": 64,
    "n_mfcc": 13,
    "resample_type": "kaiser_fast",
}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(DEVICE)


def to_float_mono(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.dtype.kind == "u":  # unsigned
        max_val = np.iinfo(x.dtype).max
        x = (x.astype(np.float32) - max_val / 2) / (max_val / 2)
    elif x.dtype.kind == "i":  # signed
        max_val = np.iinfo(x.dtype).max
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

    if raw_pcm.ndim == 2:
        raw_pcm = raw_pcm.mean(axis=1)

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

RANDOM_STATE = 42

def build_balanced_metadata(metadata_csv, audio_dir=None, queen_label_col="queen presence", samples_per_class=75000):
    df = pd.read_csv(metadata_csv)

    if "file path" in df.columns:
        df = df[df["file path"].apply(lambda p: Path(p).exists())].copy()
    elif "file name" in df.columns:
        if audio_dir is None:
            raise ValueError("audio_dir must be provided when only 'file name' is present")
        df["file path"] = df["file name"].apply(lambda s: str(Path(audio_dir) / s))
        df = df[df["file path"].apply(lambda p: Path(p).exists())].copy()
    else:
        raise KeyError(f"{metadata_csv} must contain 'file path' or 'file name'")

    class_counts = df[queen_label_col].value_counts().to_dict()
    print(class_counts)
    print(f"[INFO] Class counts before sampling: {class_counts}")
    n0 = class_counts[1]
    n1 = class_counts[0]
    N = min(n0,n1)
    if len(class_counts) < 2:
        raise RuntimeError("[STOP] Both queen (1) and no_queen (0) must be present.")

    min_count = min(df[queen_label_col].value_counts().min(), samples_per_class)
    balanced_df = (
        df.groupby(queen_label_col, group_keys=False)
          .sample(n=N, random_state=RANDOM_STATE)
          .loc[:, ["file path", queen_label_col]]
          .reset_index(drop=True)
    )
    print(f"[INFO] Class counts after sampling: {balanced_df[queen_label_col].value_counts().to_dict()}")
    return balanced_df, N
def build_dataset_from_metadata_windowed(
    metadata_csv: Path,
    win_s: float = 5.0,
    hop_s: float = 2.5,
    trim_db: float = 30.0,  # (unused, kept to avoid logic change)
    path_col: str = "file path",
    label_col: str = "queen presence",
):
    df = pd.read_csv(metadata_csv)
    X, y, groups = [], [], []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Files"):
        p = Path(row[path_col])
        label = int(row[label_col])
        try:
            sr, raw = wavfile.read(str(p), mmap=True)
            Xw, _ = extract_features_windows_memmap(raw, sr, win_s=win_s, hop_s=hop_s)
            if Xw.size == 0:
                continue
            MAX_WINS = 800
            if Xw.shape[0] > MAX_WINS:
                Xw = Xw[:MAX_WINS]

            #print(f"{p.name}: {Xw.shape[0]} windows")
            X.append(Xw)
            y.append(np.full((Xw.shape[0],), label, dtype=np.int8))
            groups.extend([p.as_posix()] * Xw.shape[0])
        except Exception as e:
            print(f"[SKIP] {p} - {e}")

    if not X:
        raise RuntimeError(f"No usable rows in {metadata_csv}")
    return np.vstack(X), np.concatenate(y), np.array(groups)

def train_from_metadata_windowed(metadata_csv: Path, win_s=2.5, hop_s=1.25):
    print(f"\nBuilding windowed dataset from {metadata_csv.resolve()}")
    X, y, groups = build_dataset_from_metadata_windowed(metadata_csv, win_s=win_s, hop_s=hop_s)
    print(f"Dataset size (windows): {X.shape[0]} samples, feature dim: {X.shape[1]}")

    # group-aware train/test split (by file)
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

    from collections import Counter
    print("Train window labels:", Counter(y_train))
    print("Test  window labels:", Counter(y_test))

    # Baseline model (single fit with default params)
    baseline_params = dict(hidden_dim=128, dropout=0.2, lr=1e-3, weight_decay=1e-4, epochs=12, batch_size=1024)
    baseline_model, _ = train_one_model(
        X_train, y_train.astype(np.float32), group_train,
        X_test,  y_test.astype(np.float32),
        in_dim=X.shape[1], **baseline_params
    )

    # Evaluate baseline
    baseline_model.eval()
    with torch.no_grad():
        logits = baseline_model(torch.from_numpy(X_test).float().to(DEVICE))
        y_pred = (torch.sigmoid(logits) >= 0.5).long().cpu().numpy()

    print("\nBaseline accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred, labels=[0,1], digits=3, zero_division=0))

    # Group K-fold CV on training groups (like original logic)
    unique_groups = np.unique(group_train).size
    n_splits = min(5, unique_groups)
    if n_splits < 2:
        print("[WARN] <2 training groups; skipping CV and using the baseline model.")
        tuned_model = baseline_model
        best_params = baseline_params
    else:
        param_grid = [
            dict(hidden_dim=128, dropout=0.2, lr=1e-3,  weight_decay=1e-4, epochs=12, batch_size=1024),
            dict(hidden_dim=256, dropout=0.3, lr=1e-3,  weight_decay=1e-4, epochs=15, batch_size=1024),
            dict(hidden_dim=128, dropout=0.1, lr=5e-4,  weight_decay=5e-5, epochs=18, batch_size=1024),
            dict(hidden_dim=256, dropout=0.2, lr=2e-3,  weight_decay=1e-4, epochs=12, batch_size=1024),
        ]
        tuned_model, best_params, best_cv = group_kfold_cv_torch(X_train, y_train, group_train, param_grid, n_splits=n_splits)
        print("Best params:", best_params)
        print("CV best score:", best_cv)

    # Final test accuracy with tuned model
    tuned_model.eval()
    with torch.no_grad():
        test_logits = tuned_model(torch.from_numpy(X_test).float().to(DEVICE))
        test_pred = (torch.sigmoid(test_logits) >= 0.5).long().cpu().numpy()
    print("Test accuracy (tuned):", accuracy_score(y_test, test_pred))

    # Save checkpoint
    torch.save({
        "model_state_dict": tuned_model.state_dict(),
        "config": FEATURE_CONFIG,
        "input_dim": X.shape[1],
        "hparams": best_params,
        "device": str(DEVICE),
    }, MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")

    return tuned_model, {"y_test": y_test, "y_pred": y_pred}


# --- Ensure your checkpoint path is correct
MODEL_PATH = "./bee_audio_torch.pt"   # fix typo if you had "torh" before
class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim=128, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)  # logits


def train_one_model(X_train, y_train, groups_train, X_val, y_val, *,
                    in_dim, hidden_dim=128, dropout=0.2,
                    lr=1e-3, weight_decay=1e-4, epochs=10, batch_size=512):
    model = MLP(in_dim, hidden_dim=hidden_dim, dropout=dropout).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    X_train_t = torch.from_numpy(X_train).float().to(DEVICE)
    y_train_t = torch.from_numpy(y_train).float().to(DEVICE)
    X_val_t   = torch.from_numpy(X_val).float().to(DEVICE)
    y_val_t   = torch.from_numpy(y_val).float().to(DEVICE)

    # Simple class weighting (similar spirit to class_weight="balanced")
    pos_weight = None
    p = y_train.mean()
    if 0 < p < 1:
        pos_weight = torch.tensor([(1 - p) / p], device=DEVICE)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Mini-batch loop
    num_train = X_train_t.shape[0]
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(num_train, device=DEVICE)
        for i in range(0, num_train, batch_size):
            idx = perm[i:i+batch_size]
            xb = X_train_t[idx]
            yb = y_train_t[idx]
            logits = model(xb)
            loss = criterion(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Validation accuracy
    model.eval()
    with torch.no_grad():
        val_logits = model(X_val_t)
        val_pred = (torch.sigmoid(val_logits) >= 0.5).long().cpu().numpy()
    val_acc = (val_pred == y_val.astype(np.int64)).mean()
    return model, val_acc

def predict_file_windowed_torch(ckpt_path, wav_path, win_s=FEATURE_CONFIG["win_s"], hop_s=FEATURE_CONFIG["hop_s"]):
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    input_dim = ckpt["input_dim"]
    hparams = ckpt.get("hparams", dict(hidden_dim=128, dropout=0.2))
    model = MLP(input_dim, hidden_dim=hparams.get("hidden_dim",128), dropout=hparams.get("dropout",0.2))
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(DEVICE).eval()

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

    Xw = np.vstack(Xw).astype(np.float32)
    with torch.no_grad():
        logits = model(torch.from_numpy(Xw).to(DEVICE))
        probs = torch.sigmoid(logits).cpu().numpy()
        pred  = (probs >= 0.5).astype(np.int64)

    return {
        "frame_pred": pred,
        "frame_proba": probs,
        "intervals": intervals,
        "maj_vote": int(pred.mean() >= 0.5),
        "mean_prob": float(probs.mean()),
    }


# -----------------------------
# CONFIGURE PATHS (edit these)
# -----------------------------
ROOT = Path(r"C:\Users\Lisa\Desktop\BeeWinged")

AUDIO_DIRS = {
    "TBON":   ROOT / "TBON"   / "TBON",
    "SBCM":   ROOT / "SBCM"   / "SBCM",
    "NUHIVE": ROOT / "NUHIVE" / "NUHIVE",
    "BAD":    ROOT / "BAD"    / "BAD",
}

# Point to your four .h5 files that contain columns: file name, queen presence
H5_FILES = {
    "TBON":   ROOT / "TBON"   / "TBON_labels.h5",
    "SBCM":   ROOT / "SBCM"   / "SBCM_labels.h5",
    "NUHIVE": ROOT / "NUHIVE" / "NUHIVE_labels.h5",
    "BAD":    ROOT / "BAD"    / "BAD_labels.h5",
}

# -----------------------------
# HDF5 loader (robust to different keys)
# -----------------------------
def _read_labels_h5(h5_path: Path) -> pd.DataFrame:
    """
    Returns a DataFrame with columns ['file name','queen presence'].
    Scans all HDF5 keys and picks the first table that contains both columns.
    """
    with pd.HDFStore(str(h5_path), mode="r") as store:
        keys = store.keys()
        for k in keys:
            df = store.select(k)
            cols_lower = {c.lower() for c in df.columns}
            if {"file name", "queen presence"}.issubset(cols_lower):
                df = df.rename(columns={c: c.lower() for c in df.columns})
                df = df[["file name", "queen presence"]].copy()
                df = df.dropna()
                return df
            
    raise ValueError(f"Could not find ['file name','queen presence'] in {h5_path} (keys={keys})")
def group_kfold_cv_torch(X, y, groups, param_grid, n_splits):
    # Manual CV over hyperparameters (mirrors GridSearchCV + GroupKFold idea from sklearn)
    best_score = -np.inf
    best_params = None
    best_model = None

    gkf = GroupKFold(n_splits=n_splits)
    in_dim = X.shape[1]

    for params in param_grid:
        scores = []
        for tr_idx, va_idx in gkf.split(X, y, groups=groups):
            X_tr, X_va = X[tr_idx], X[va_idx]
            y_tr, y_va = y[tr_idx].astype(np.float32), y[va_idx].astype(np.float32)
            groups_tr = groups[tr_idx]

            model, val_acc = train_one_model(
                X_tr, y_tr, groups_tr, X_va, y_va,
                in_dim=in_dim,
                hidden_dim=params.get("hidden_dim", 128),
                dropout=params.get("dropout", 0.2),
                lr=params.get("lr", 1e-3),
                weight_decay=params.get("weight_decay", 1e-4),
                epochs=params.get("epochs", 10),
                batch_size=params.get("batch_size", 512),
            )
            scores.append(val_acc)

        mean_score = float(np.mean(scores))
        print(f"Params {params} -> CV mean acc = {mean_score:.4f}")
        if mean_score > best_score:
            best_score = mean_score
            best_params = params
            # Retrain on full data with these params to get a model snapshot
            best_model, _ = train_one_model(
                X, y.astype(np.float32), groups,
                X, y.astype(np.float32),
                in_dim=in_dim,
                hidden_dim=params.get("hidden_dim", 128),
                dropout=params.get("dropout", 0.2),
                lr=params.get("lr", 1e-3),
                weight_decay=params.get("weight_decay", 1e-4),
                epochs=params.get("epochs", 10),
                batch_size=params.get("batch_size", 512),
            )

    return best_model, best_params, best_score

def build_labels_csv_from_h5s(audio_dirs: dict, h5_files: dict, out_csv: Path) -> Path:
    """
    Builds a single CSV with columns: file path, queen presence, dataset
    by joining each H5 'file name' to its corresponding audio folder.
    Drops rows whose file path does not exist.
    """
    frames = []
    for ds_name, h5_path in h5_files.items():
        if ds_name not in audio_dirs:
            print(f"[WARN] No audio dir mapping for dataset '{ds_name}', skipping.")
            continue
        if not Path(h5_path).exists():
            print(f"[WARN] Missing H5 file for {ds_name}: {h5_path}")
            continue

        df = _read_labels_h5(Path(h5_path))
        df["queen presence"] = df["queen presence"].astype(int)

        base = Path(audio_dirs[ds_name])
        def resolve_path(fn):
            p = Path(fn)
            if p.is_absolute() and p.exists():
                return str(p.resolve())
            return str((base / fn).resolve())

        df["file path"] = df["file name"].apply(resolve_path)
        df["dataset"]   = ds_name
        frames.append(df[["file path", "queen presence", "dataset"]])

    if not frames:
        raise RuntimeError("No labeled rows loaded from H5 files.")

    all_df = pd.concat(frames, ignore_index=True).drop_duplicates("file path")
    all_df["exists"] = all_df["file path"].apply(lambda p: Path(p).exists())
    missing = int((~all_df["exists"]).sum())
    if missing:
        print(f"[INFO] Dropping {missing} rows whose file path does not exist on disk.")
    all_df = all_df[all_df["exists"]].drop(columns=["exists"])

    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    all_df.to_csv(out_csv, index=False)
    print(f"[INFO] Wrote {len(all_df)} labeled rows to {out_csv}")
    print(all_df.groupby(["dataset","queen presence"]).size())
    return out_csv

# -----------------------------
# Train using your existing trainer on the generated CSV
# -----------------------------
def train_from_h5s(h5_files: dict,
                   audio_dirs: dict,
                   *,
                   win_s: float = 2.0,
                   hop_s: float = 2.0,
                   balance: bool = True,
                   samples_per_class: int = 75000,
                   out_prefix: str = "h5_all",
                   model_path: str = MODEL_PATH):
    """
    1) Build a unified CSV (file path, queen presence, dataset) from the H5s
    2) (Optional) balance the file list per class using your existing helper
    3) Train with windowed features using your existing train_from_metadata_windowed
    """

    # 1) Build CSV
    meta_csv = build_labels_csv_from_h5s(audio_dirs, h5_files, out_csv=f"{out_prefix}_labels.csv")

    # 2) Balance at the FILE level (reuses your build_balanced_metadata)
    csv_for_train = meta_csv
    if balance:
        global N
        balanced_df, N = build_balanced_metadata(meta_csv, queen_label_col="queen presence",
                                              samples_per_class=samples_per_class)
        balanced_csv = Path(f"{out_prefix}_labels_balanced.csv")
        balanced_df.to_csv(balanced_csv, index=False)
        print(f"[INFO] Balanced CSV saved to {balanced_csv} "
              f"({balanced_df['queen presence'].value_counts().to_dict()})")
        csv_for_train = balanced_csv

    # 3) Train (uses your existing pipeline & saves checkpoint to MODEL_PATH)
    print("\n[TRAIN] Starting training with win_s="
          f"{win_s}, hop_s={hop_s} on {csv_for_train}")
    model, eval_dict = train_from_metadata_windowed(
        Path(csv_for_train),
        win_s=win_s,
        hop_s=hop_s
    )

    # Overwrite the default save with our preferred name if needed
    if model_path != "bee_audio_torch.pt":
        torch.save({
            "model_state_dict": model.state_dict(),
            "config": FEATURE_CONFIG,
            "input_dim": next(model.net.children()).in_features,
            "hparams": {"hidden_dim": list(model.net.children())[0].out_features,
                        "dropout": list(model.net.children())[2].p},
            "device": str(DEVICE),
        }, model_path)
        print(f"[INFO] Saved model to {model_path}")

    return model, eval_dict, csv_for_train

model, eval_dict, used_csv = train_from_h5s(
    h5_files=H5_FILES,
    audio_dirs=AUDIO_DIRS,
    win_s=2.0, hop_s=2.0,           # 2-second clips
    balance=True,                   # balance file counts per class
    samples_per_class=N,       # upper cap; min(existing, this) is used
    out_prefix="h5_all",            # outputs: h5_all_labels.csv, h5_all_labels_balanced.csv
    model_path=MODEL_PATH           # "./bee_audio_torch_all.pt"
)