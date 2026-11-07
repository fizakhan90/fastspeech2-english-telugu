import pathlib, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

WS = pathlib.Path.home() / "fs2"
LANGS = {"Svarah": "English", "TeluguMale": "Telugu"}
KEYS = ["total_loss", "mel_loss", "mel_postnet_loss", "pitch_loss", "energy_loss", "duration_loss"]

for name, lang in LANGS.items():
    out = WS / "results" / lang
    out.mkdir(parents=True, exist_ok=True)
    dfs = {}
    for split in ["train", "val"]:
        ea = EventAccumulator(str(WS / "output/log" / name / split), size_guidance={"scalars": 0})
        ea.Reload()
        cols = {}
        for k in KEYS:
            tag = f"Loss/{k}"
            if tag in ea.Tags()["scalars"]:
                cols[k] = pd.Series({e.step: e.value for e in ea.Scalars(tag)})
        df = pd.DataFrame(cols).sort_index()
        df.index.name = "step"
        df.to_csv(out / f"{split}_losses.csv")
        dfs[split] = df

        fig, axes = plt.subplots(2, 3, figsize=(14, 7))
        for ax, k in zip(axes.flat, KEYS):
            s = df[k].rolling(10, min_periods=1).mean() if split == "train" else df[k]
            ax.plot(s.index, s.values)
            ax.set_title(k); ax.set_xlabel("step"); ax.grid(alpha=.3)
        fig.suptitle(f"{lang} – {'training' if split=='train' else 'validation'} loss")
        fig.tight_layout()
        fig.savefig(out / f"{'training' if split=='train' else 'validation'}_loss.png", dpi=150)
        plt.close(fig)

    v = dfs["val"]
    ck = v[v.index % 5000 == 0]          # only these steps have checkpoints
    print(f"\n== {lang}: {len(dfs['train'])} train / {len(v)} val points")
    print(f"  min val total: step {v.total_loss.idxmin()} ({v.total_loss.min():.4f})")
    print(f"  best saved ckpt (val total): step {ck.total_loss.idxmin()} ({ck.total_loss.min():.4f})")
    print(f"  best saved ckpt (val mel):   step {ck.mel_loss.idxmin()} ({ck.mel_loss.min():.4f})")
    print(ck[["total_loss", "mel_loss", "pitch_loss", "energy_loss", "duration_loss"]]
          .iloc[::2].round(3).to_string())