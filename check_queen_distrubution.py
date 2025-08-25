from pathlib import Path
p = Path("C:/Users/ezetaxe/PyCharmMiscProject/beewinged/all_audio_files/NUHIVE/NUHIVE")
names = [f.name for f in p.glob("*.wav")]
counts = {
    "queen": sum(("QueenBee" in n) and ("NO_QueenBee" not in n) for n in names),
    "no_queen": sum("NO_QueenBee" in n for n in names),
}
print(counts)
