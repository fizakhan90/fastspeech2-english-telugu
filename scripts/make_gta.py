import sys, pathlib, yaml, json, numpy as np, torch, tgt, librosa
from types import SimpleNamespace
from scipy.io import wavfile

WS = pathlib.Path.home() / "fs2"
sys.path.insert(0, str(WS / "FastSpeech2"))
from text import text_to_sequence
from utils.model import get_model
dev = torch.device("cuda")
SIL = {"", "sil", "sp", "spn"}

for name in ["Svarah", "TeluguMale"]:
    cfg = [yaml.safe_load(open(WS / f"FastSpeech2/config/{name}/{k}.yaml")) for k in ["preprocess", "model", "train"]]
    pp, mc, tc = cfg
    pdir, rdir = pathlib.Path(pp["path"]["preprocessed_path"]), pathlib.Path(pp["path"]["raw_path"])
    spk_map = json.load(open(pdir / "speakers.json"))
    model = get_model(SimpleNamespace(restore_step=25000), cfg, dev, train=False)
    hop, sr = pp["preprocessing"]["stft"]["hop_length"], pp["preprocessing"]["audio"]["sampling_rate"]
    out = WS / "gta" / name
    (out / "mels").mkdir(parents=True, exist_ok=True); (out / "wavs").mkdir(exist_ok=True)
    T = lambda x, dt=torch.float: torch.from_numpy(np.asarray(x)).to(dt).unsqueeze(0).to(dev)

    for split in ["train", "val"]:
        ids, l1 = [], []
        for line in open(pdir / f"{split}.txt", encoding="utf-8").read().splitlines():
            b, spk, ph, _ = line.split("|")
            ld = lambda k: np.load(pdir / k / f"{spk}-{k}-{b}.npy")
            mel, pit, eng, dur = ld("mel"), ld("pitch"), ld("energy"), ld("duration")
            n = int(dur.sum())

            # trim wav exactly like ming024: start at first non-silence phone, length = n*hop
            tier = tgt.io.read_textgrid(pdir / "TextGrid" / spk / f"{b}.TextGrid",
                                        include_empty_intervals=True).get_tier_by_name("phones")
            start = next(t.start_time for t in tier._objects if t.text not in SIL)
            wav, _ = librosa.load(rdir / spk / f"{b}.wav", sr=sr)
            seg = wav[int(sr * start): int(sr * start) + n * hop]
            seg = np.pad(seg, (0, n * hop - len(seg)))

            seq = np.array(text_to_sequence(ph, pp["preprocessing"]["text"]["text_cleaners"]))
            with torch.no_grad():
                o = model(torch.tensor([spk_map[spk]], device=dev), T(seq, torch.long),
                          torch.tensor([len(seq)], device=dev), len(seq),
                          mels=T(mel), mel_lens=torch.tensor([n], device=dev), max_mel_len=n,
                          p_targets=T(pit), e_targets=T(eng), d_targets=T(dur, torch.long))
            gta = o[1][0, :n].cpu().numpy()                       # (n, 80)
            l1.append(np.abs(gta - mel[:n]).mean())

            np.save(out / "mels" / f"{b}.npy", gta.T.astype(np.float32))   # HiFi-GAN wants (80, T)
            wavfile.write(out / "wavs" / f"{b}.wav", sr, (seg * 32767).clip(-32768, 32767).astype(np.int16))
            ids.append(b)
        open(out / f"{split}.txt", "w").write("\n".join(f"{i}|" for i in ids) + "\n")
        print(f"{name} {split}: {len(ids)} clips | mean L1(GTA mel, real mel) = {np.mean(l1):.4f}")