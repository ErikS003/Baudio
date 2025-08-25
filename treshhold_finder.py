import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support

CSV = r"C:\Users\ezetaxe\PyCharmMiscProject\beewinged\all_audio_files\TBON\eval_results_per_file_20250821_145014.csv"
df = pd.read_csv(CSV)

y = df["true_label"].astype(int).values
scores = df["mean_prob"].values           # continuous scores
pos_frac = df["pos_win_frac"].values      # fraction of windows predicted positive (at 0.5 window-thr in your run)

def metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    pr, rc, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    tpr = tp / (tp + fn) if (tp + fn) else 0.0
    tnr = tn / (tn + fp) if (tn + fp) else 0.0
    bal = 0.5 * (tpr + tnr)
    return {"acc":acc, "precision":pr, "recall":rc, "f1":f1, "balanced_acc":bal, "tn":tn, "fp":fp, "fn":fn, "tp":tp}

# -------- sweep mean_prob threshold only --------
ths = np.linspace(0.05, 0.95, 91)
rows = []
for t in ths:
    yhat = (scores >= t).astype(int)
    m = metrics(y, yhat); m["thr"] = t; rows.append(m)
res = pd.DataFrame(rows)

print("\nTop 5 thresholds by BALANCED ACCURACY:")
print(res.sort_values("balanced_acc", ascending=False).head(5)[["thr","balanced_acc","acc","precision","recall","f1","tn","fp","fn","tp"]])

print("\nTop 5 thresholds by F1:")
print(res.sort_values("f1", ascending=False).head(5)[["thr","balanced_acc","acc","precision","recall","f1","tn","fp","fn","tp"]])

# Example: pick the best balanced-accuracy threshold
best_bal = res.sort_values("balanced_acc", ascending=False).iloc[0]
t_best = float(best_bal["thr"])
print(f"\nChosen threshold (balanced acc): t={t_best:.2f} | bal_acc={best_bal['balanced_acc']:.3f} "
      f"acc={best_bal['acc']:.3f} prec={best_bal['precision']:.3f} rec={best_bal['recall']:.3f}")

# Optional: enforce a precision target (fewer false alarms)
TARGET_PREC = 0.80
cand = res[res["precision"] >= TARGET_PREC]
if not cand.empty:
    pick = cand.sort_values("recall", ascending=False).iloc[0]
    print(f"Threshold meeting precision ≥{TARGET_PREC:.2f}: t={pick['thr']:.2f} "
          f"(prec={pick['precision']:.3f}, rec={pick['recall']:.3f}, acc={pick['acc']:.3f}, bal_acc={pick['balanced_acc']:.3f})")
else:
    print(f"No threshold reaches precision ≥{TARGET_PREC:.2f} with current scores.")

# Save predictions with the chosen threshold
df["pred_t_custom"] = (scores >= t_best).astype(int)
out = CSV.replace(".csv", f"_with_pred_t{t_best:.2f}.csv")
df.to_csv(out, index=False)
print("Wrote:", out)
