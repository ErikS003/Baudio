import pandas as pd
path_to_dataset = r"./data/archive/"

NUHIVE = "NUHIVE.h5"
TBON = "TBON.h5"
SBCM = "SBCM.h5"
BAD = "BAD.h5"

df_SBCM = pd.read_hdf(path_to_dataset + SBCM, 'bee_audio')
df_NUHIVE = pd.read_hdf(path_to_dataset + NUHIVE, 'bee_audio')
df_TBON = pd.read_hdf(path_to_dataset + TBON, 'bee_audio')
df_BAD = pd.read_hdf(path_to_dataset + BAD, 'bee_audio')

df_concat = pd.concat([df_NUHIVE, df_SBCM, df_TBON, df_BAD], ignore_index=True)
print(df_concat.head())

Q_1 = Q_0 = Q_nan = 0

for i in range(len(df_concat["queen presence"])):
    if df_concat["queen presence"][i]==1:
        Q_1+=1
    elif df_concat["queen presence"][i]:
        Q_0+=1
    else:
        Q_nan+=1

print(f"queen present: {Q_1}",f"\n queen not present: {Q_0}",f"\n not recorded: {Q_nan}\n", len(df_concat["queen presence"]))
columns=df_concat.columns
drops = []
for i in range(len(columns)):
    if columns[i]!="queen presence" and columns[i]!="file name":
        drops.append(columns[i])
    
df_queen = df_concat.drop(columns=drops)
df_queen = df_queen.dropna()
print(df_queen)
#for k in range(len(df_queen["queen presence"])):
#    print(df_queen["file name"][k])
import matplotlib.pyplot as plt

HIVE1 = df_queen.where(df_queen["file name"].str.contains("Hive1"))
HIVE2 = df_queen.where(df_queen["file name"].str.contains("Hive2"))
HIVE3 = df_queen.where(df_queen["file name"].str.contains("Hive3"))

print(HIVE1.head())