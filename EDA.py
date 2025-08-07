import os, glob, re
from datetime import datetime
import pandas as pd

annotations_path = r"./data/annotations/inspections_2022.csv"
audio_path       = r"./data/audio/beehives_2022/audio_2022_chunk_1"

# ---------------------------------------------------
# 1) Build wav_df
pattern = re.compile(r"(\d{2})-(\d{2})-(\d{4})_(\d{2})h(\d{2})_HIVE-(\d+)\.wav")
records = []

for p in glob.glob(os.path.join(audio_path, "*.wav")):
    fn = os.path.basename(p)
    m = pattern.match(fn)
    if not m:
        print(f"⚠ Skipping {fn} (pattern mismatch)")
        continue
    dd, mm, yyyy, hh, mins, tag = m.groups()
    ts = datetime.strptime(f"{yyyy}-{mm}-{dd} {hh}:{mins}", "%Y-%m-%d %H:%M")
    records.append({
        "wav_path": p,
        "wav_filename": fn,
        "recording_datetime": ts,
        "tag_number": int(tag)
    })

wav_df = (
    pd.DataFrame(records)
      .sort_values(["recording_datetime", "tag_number"], ignore_index=True)
)

# Localize to UTC
wav_df["recording_datetime"] = wav_df["recording_datetime"].dt.tz_localize("UTC")

# ---------------------------------------------------
# 2) Load annotations and localize
ann = (
    pd.read_csv(annotations_path, parse_dates=["Date"])
      .rename(columns={"Date": "insp_datetime", "Tag number": "tag_number"})
      .assign(tag_number=lambda df: df["tag_number"].astype(int))
)

if ann["insp_datetime"].dt.tz is None:
    ann["insp_datetime"] = ann["insp_datetime"].dt.tz_localize("UTC")
else:
    ann["insp_datetime"] = ann["insp_datetime"].dt.tz_convert("UTC")

ann = ann.sort_values(["insp_datetime", "tag_number"], ignore_index=True)

# ---------------------------------------------------
# ---- 3) merge ------------------------------------------------------------
merged = pd.merge_asof(
    wav_df,
    ann,
    by="tag_number",
    left_on="recording_datetime",
    right_on="insp_datetime",
    direction="nearest",
    tolerance=pd.Timedelta("7 days"),
)
# ---------------------------------------------------
# 4) Save
out_fname = "wav_with_annotations.csv"
merged.to_csv(out_fname, index=False)
print(f"Merged table written to {out_fname} ({len(merged)} rows)")
