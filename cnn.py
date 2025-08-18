# cnn_imagenet_from_spectrograms.py
import os, ast, re, difflib, math, random
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import models, transforms
from PIL import Image

import librosa
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from joblib import dump

# -------------------- Config --------------------
CSV_PATH = "rms_presence_pairs.csv"    # <-- your CSV
WANTED_LABEL = "queen_presence"        # <-- your label column
SR = 8000
DURATION_S = 2.0
KNOWN_LEN = int(SR * DURATION_S)       # 16000
SUBSET_FRAC = 0.1                      # use full data for CNN (1.0) if possibl
RANDOM_STATE = 42

# Spectrogram
N_FFT = 512
HOP = 128
N_MELS = 64
FMIN, FMAX = 20, 4000
DB_TOP_DB = 80.0

# Model / training
BACKBONE = "resnet18"                  # resnet18 / resnet34 / efficientnet_b0
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 20
FREEZE_EPOCHS = 3                      # freeze backbone for first few epochs
LR = 1e-3
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 4

SAVE_DIR = Path("artifacts")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# -------------------- Helpers (parsers) --------------------
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
    return np.fromstring(cleaned, sep=" ", dtype=np.float32).ravel()

def fix_length(x, L):
    if len(x) == L: return x
    if len(x) > L:  return x[:L]
    out = np.zeros(L, dtype=np.float32); out[:len(x)] = x; return out

def find_col(df, wanted):
    df = df.rename(columns={c: c.strip() for c in df.columns})
    cols_lower = {c.lower(): c for c in df.columns}
    if wanted.lower() in cols_lower:
        return cols_lower[wanted.lower()]
    match = difflib.get_close_matches(wanted.lower(), cols_lower.keys(), n=1, cutoff=0.8)
    if match:
        return cols_lower[match[0]]
    raise KeyError(f"Couldn't find label column {wanted!r}. Available: {list(df.columns)}")

# -------------------- Audio -> spectrogram --------------------
def series_to_logmel_img(
    x: np.ndarray,
    sr: int = SR,
    n_fft: int = N_FFT,
    hop: int = HOP,
    n_mels: int = N_MELS,
    fmin: int = FMIN,
    fmax: int = FMAX,
    top_db: float = DB_TOP_DB,
) -> np.ndarray:
    # Normalize RMS if you did so before; comment out if amplitude carries signal
    if x.size:
        rms = float(np.sqrt(np.mean(x**2)))
        if rms > 0:
            x = x / rms

    S = librosa.feature.melspectrogram(
        y=x, sr=sr, n_fft=n_fft, hop_length=hop,
        n_mels=n_mels, fmin=fmin, fmax=fmax, power=2.0,
        window="hann", center=True
    )
    S_db = librosa.power_to_db(S, ref=np.max, top_db=top_db)  # (n_mels, T) in dB
    # Min-max to [0,1] (dataset standardization happens later)
    S_min, S_max = S_db.min(), S_db.max()
    if S_max > S_min:
        S_norm = (S_db - S_min) / (S_max - S_min)
    else:
        S_norm = np.zeros_like(S_db)
    return S_norm.astype(np.float32)

def spec_to_pil(spec: np.ndarray, img_size: int = IMG_SIZE) -> Image.Image:
    img = (spec * 255.0).clip(0, 255).astype(np.uint8)
    pil = Image.fromarray(img)  # grayscale
    w, h = pil.size
    side = max(w, h)
    canvas = Image.new("L", (side, side), color=0)
    canvas.paste(pil, ((side - w)//2, (side - h)//2))
    # Make it RGB here so transforms don’t need a Lambda
    return canvas.resize((img_size, img_size), Image.BICUBIC).convert("RGB")

# -------------------- Dataset --------------------
class SpectrogramDataset(Dataset):
    def __init__(self, df: pd.DataFrame, label_col: str, classes: List[str], augment: bool):
        self.df = df.reset_index(drop=True)
        self.label_col = label_col
        self.classes = classes
        self.class_to_idx = {c:i for i,c in enumerate(classes)}
        self.augment = augment

        # Vision transforms (normalize to ImageNet)
        self.tf = transforms.Compose([
            transforms.ToTensor(),  # now returns shape (3,H,W) because image is RGB
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        y_str = str(row[self.label_col])
        label = self.class_to_idx[y_str]

        x = parse_series(row["rms"])
        x = fix_length(x, KNOWN_LEN)

        # Optional waveform augmentations (small & safe)
        if self.augment:
            if random.random() < 0.5:
                # small random time shift (wrap)
                shift = random.randint(-int(0.05*KNOWN_LEN), int(0.05*KNOWN_LEN))
                x = np.roll(x, shift)
            if random.random() < 0.3:
                # tiny Gaussian noise
                x = x + np.random.randn(*x.shape).astype(np.float32) * 0.005

        spec = series_to_logmel_img(x)

        # SpecAugment (time/freq masking) on the spectrogram
        if self.augment:
            spec = self._spec_augment(spec, max_time_masks=2, max_freq_masks=2)

        pil = spec_to_pil(spec, IMG_SIZE)
        img = self.tf(pil)
        return img, label

    @staticmethod
    def _spec_augment(spec: np.ndarray, max_time_masks=2, max_freq_masks=2):
        s = spec.copy()
        n_mels, T = s.shape
        for _ in range(random.randint(0, max_freq_masks)):
            f = random.randint(0, max(1, n_mels//8))
            f0 = random.randint(0, max(0, n_mels - f))
            s[f0:f0+f, :] = 0.0
        for _ in range(random.randint(0, max_time_masks)):
            t = random.randint(0, max(1, T//8))
            t0 = random.randint(0, max(0, T - t))
            s[:, t0:t0+t] = 0.0
        return s

# -------------------- Model --------------------
def build_model(n_classes: int, backbone: str = BACKBONE):
    if backbone == "resnet18":
        m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        in_features = m.fc.in_features
        m.fc = nn.Linear(in_features, n_classes)
    elif backbone == "resnet34":
        m = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        in_features = m.fc.in_features
        m.fc = nn.Linear(in_features, n_classes)
    elif backbone == "efficientnet_b0":
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        in_features = m.classifier[1].in_features
        m.classifier[1] = nn.Linear(in_features, n_classes)
    else:
        raise ValueError("Unsupported backbone")
    return m

# -------------------- Training utils --------------------
def seed_everything(seed=RANDOM_STATE):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def compute_sample_weights(y: np.ndarray) -> np.ndarray:
    vals, counts = np.unique(y, return_counts=True)
    count_map = {v:c for v,c in zip(vals, counts)}
    return np.array([1.0 / count_map[yy] for yy in y], dtype=np.float32)

# -------------------- Main --------------------
def main():
    seed_everything()

    df_full = pd.read_csv(CSV_PATH)
    label_col = find_col(df_full, WANTED_LABEL)

    if "rms" not in df_full.columns:
        raise KeyError("CSV must have an 'rms' column with the time-series.")

    # optional subset (for quick tests)
    if SUBSET_FRAC < 1.0:
        y_str = df_full[label_col].astype(str).to_numpy()
        idx_train, idx_subset = train_test_split(
            np.arange(len(df_full)),
            test_size=SUBSET_FRAC,
            random_state=RANDOM_STATE,
            stratify=y_str
        )
        df = df_full.iloc[idx_subset].reset_index(drop=True)
    else:
        df = df_full.reset_index(drop=True)

    # drop rows with empty rms
    keep = df["rms"].apply(lambda x: len(parse_series(x)) > 0).to_numpy()
    if (~keep).any():
        print(f"Dropping {(~keep).sum()} rows with empty 'rms'.")
        df = df.loc[keep].reset_index(drop=True)

    # split
    y_all = df[label_col].astype(str).to_numpy()
    classes = sorted(np.unique(y_all).tolist())
    X_train, X_val, y_train, y_val = train_test_split(
        df, y_all, test_size=0.2, random_state=RANDOM_STATE, stratify=y_all
    )

    # datasets
    ds_tr = SpectrogramDataset(X_train, label_col, classes, augment=True)
    ds_va = SpectrogramDataset(X_val, label_col, classes, augment=False)

    # sampler for imbalance
    y_tr_idx = np.array([ds_tr.class_to_idx[str(r[label_col])] for _, r in X_train.iterrows()])
    weights = compute_sample_weights(y_tr_idx)
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    # loaders
    dl_tr = DataLoader(ds_tr, batch_size=BATCH_SIZE, sampler=sampler, num_workers=NUM_WORKERS, pin_memory=True)
    dl_va = DataLoader(ds_va, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(n_classes=len(classes), backbone=BACKBONE).to(device)

    # loss
    if len(classes) == 2:
        # y in {0,1}, use BCEWithLogits
        pos_count = (y_tr_idx == 1).sum()
        neg_count = (y_tr_idx == 0).sum()
        pos_weight = torch.tensor([neg_count / max(1, pos_count)], dtype=torch.float32).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        # multi-class
        class_counts = np.bincount(y_tr_idx, minlength=len(classes))
        class_weights = (class_counts.sum() / np.maximum(1, class_counts)).astype(np.float32)
        criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32).to(device))

    # optimizer & scheduler
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(params, lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # freeze backbone for a few epochs
    def set_backbone_requires_grad(req: bool):
        if isinstance(model, models.ResNet):
            for name, p in model.named_parameters():
                if not name.startswith("fc."):
                    p.requires_grad = req
        else:
            for name, p in model.named_parameters():
                if "classifier.1" not in name:
                    p.requires_grad = req

    set_backbone_requires_grad(False)  # freeze initially

    best_val_f1 = -1.0
    best_state = None

    for epoch in range(1, EPOCHS+1):
        model.train()
        # unfreeze after FREEZE_EPOCHS
        if epoch == FREEZE_EPOCHS + 1:
            set_backbone_requires_grad(True)
            optimizer = optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR/5, weight_decay=WEIGHT_DECAY)
            scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS-epoch+1)

        total_loss = 0.0
        for imgs, labels in dl_tr:
            imgs = imgs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(imgs)

            if len(classes) == 2:
                # logits shape (B,2) for resnet? We set last layer to 2 for multi-class;
                # for BCE we want 1 logit. Convert if needed:
                if logits.shape[1] == 2:
                    # use logit for class 1 minus class 0 as a single logit
                    logits1 = logits[:, 1] - logits[:, 0]
                else:
                    logits1 = logits.squeeze(1)
                loss = criterion(logits1, labels.float())
            else:
                loss = criterion(logits, labels.long())

            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu())

        # validate
        model.eval()
        y_true, y_score, y_pred = [], [], []
        with torch.no_grad():
            for imgs, labels in dl_va:
                imgs = imgs.to(device)
                labels = labels.to(device)
                logits = model(imgs)
                if len(classes) == 2:
                    if logits.shape[1] == 2:
                        score = (logits[:,1] - logits[:,0]).cpu().numpy()
                        pred = logits.argmax(dim=1).cpu().numpy()
                    else:
                        score = logits.squeeze(1).cpu().numpy()
                        pred = (logits.squeeze(1) > 0).long().cpu().numpy()
                    y_score.extend(score.tolist())
                else:
                    pred = logits.argmax(dim=1).cpu().numpy()
                y_pred.extend(pred.tolist())
                y_true.extend(labels.cpu().numpy().tolist())

        # provisional F1 (threshold tuning after training)
        if len(classes) == 2:
            f1 = f1_score(y_true, y_pred, pos_label=1)
        else:
            f1 = f1_score(y_true, y_pred, average="macro")

        scheduler.step()
        print(f"Epoch {epoch:02d}: train_loss={total_loss/len(dl_tr):.4f} val_f1={f1:.4f}")

        if f1 > best_val_f1:
            best_val_f1 = f1
            best_state = {
                "model": model.state_dict(),
                "classes": classes,
                "backbone": BACKBONE,
                "img_size": IMG_SIZE,
            }

    # Load best
    if best_state is not None:
        model.load_state_dict(best_state["model"])

    # Final threshold tuning (binary only)
    print("\n=== Validation metrics ===")
    model.eval()
    y_true, y_score = [], []
    with torch.no_grad():
        for imgs, labels in dl_va:
            imgs = imgs.to(device)
            labels = labels.to(device)
            logits = model(imgs)
            if len(classes) == 2:
                if logits.shape[1] == 2:
                    score = (logits[:,1] - logits[:,0]).cpu().numpy()
                else:
                    score = logits.squeeze(1).cpu().numpy()
                y_score.extend(score.tolist())
            y_true.extend(labels.cpu().numpy().tolist())

    if len(classes) == 2:
        # choose minority as positive label (like your script)
        vals, counts = np.unique(y_tr_idx, return_counts=True)
        pos_label = int(vals[np.argmin(counts)])
        ths = np.quantile(np.array(y_score), np.linspace(0.05, 0.95, 61))
        best_f1, best_th = -1.0, 0.0
        for th in ths:
            y_hat = (np.array(y_score) >= th).astype(int)
            # ensure >=th corresponds to pos_label=1; if not, flip
            if pos_label == 0:
                y_hat = 1 - y_hat
            f1 = f1_score(y_true, y_hat, pos_label=1)
            if f1 > best_f1:
                best_f1, best_th = f1, th

        # Final report at tuned threshold
        y_hat = (np.array(y_score) >= best_th).astype(int)
        if pos_label == 0:
            y_hat = 1 - y_hat

        acc = accuracy_score(y_true, y_hat)
        print(f"Chosen pos_label={classes[pos_label]!r}  tuned threshold={best_th:.4f}  val F1={best_f1:.3f}")
        print("Accuracy:", acc)
        print("\nClassification report:\n", classification_report(y_true, y_hat, target_names=classes))
        print("\nConfusion matrix:\n", confusion_matrix(y_true, y_hat))
    else:
        # Multiclass report at argmax
        # Re-run forward for argmax preds
        y_pred = []
        with torch.no_grad():
            for imgs, labels in dl_va:
                logits = model(imgs.to(device))
                y_pred.extend(logits.argmax(dim=1).cpu().numpy().tolist())
        acc = accuracy_score(y_true, y_pred)
        print("Accuracy:", acc)
        print("\nClassification report:\n", classification_report(y_true, y_pred, target_names=classes))
        print("\nConfusion matrix:\n", confusion_matrix(y_true, y_pred))

    # Save
    torch.save(model.state_dict(), SAVE_DIR / "resnet_spectrogram.pth")
    dump(best_state, SAVE_DIR / "resnet_spectrogram_meta.joblib")
    print(f"\nSaved model + meta to {SAVE_DIR.resolve()}")
    print("Done.")

if __name__ == "__main__":
    main()
