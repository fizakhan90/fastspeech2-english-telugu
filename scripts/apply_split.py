import sys, pathlib, shutil
import numpy as np, pandas as pd

home = pathlib.Path.home()
WS = home / "fs2"
sys.path.insert(0, str(WS / "FastSpeech2"))
from text import text_to_sequence

HOP_S = 256 / 22050
sets = {"Svarah": ("en", ["english_cleaners"]), "TeluguMale": ("te", [])}

for name, (lang, cleaners) in sets.items():
    pdir = WS / "data/preprocessed_data" / name
    allf = pdir / "all.txt"
    if not allf.exists():
        shutil.copy(pdir / "train.txt", allf)          # keep ming024's full list
    lines = {l.split("|")[0]: l for l in open(allf, encoding="utf-8").read().splitlines() if l}

    split = {}
    for s in ["train", "val"]:
        df = pd.read_csv(WS / f"data/raw_data/{lang}_{s}.csv")
        ids = [pathlib.Path(p).stem for p in df["path"]]
        assert len(ids) == len(set(ids)), f"{name}: duplicate basenames in {s} csv"
        split[s] = ids
    assert not set(split["train"]) & set(split["val"]), f"{name}: train/val overlap!"

    print(f"\n== {name}")
    for s, ids in split.items():
        kept = [i for i in ids if i in lines]
        dropped = [i for i in ids if i not in lines]
        open(pdir / f"{s}.txt", "w", encoding="utf-8").write("\n".join(lines[i] for i in kept) + "\n")

        frames, bad_sym, raw_s = [], [], 0.0
        df = pd.read_csv(WS / f"data/raw_data/{lang}_{s}.csv")
        dur = dict(zip([pathlib.Path(p).stem for p in df["path"]], df["dur"]))
        for i in kept:
            b, spk, ph, _ = lines[i].split("|")
            phones = ph.strip("{}").split()
            if len(text_to_sequence(ph, cleaners)) != len(phones):
                bad_sym.append(i)
            frames.append(np.load(pdir / "mel" / f"{spk}-mel-{b}.npy").shape[0])
            raw_s += dur[i]
        f = np.array(frames)
        print(f"{s:5s}: {len(kept)} kept, {len(dropped)} dropped {dropped[:5]} | "
              f"raw {raw_s/3600:.2f} h, trimmed {f.sum()*HOP_S/3600:.2f} h | "
              f"max frames {f.max()} (>1500: {(f>1500).sum()}) | symbol mismatches {len(bad_sym)} {bad_sym[:3]}")