import sys, pathlib, shutil, yaml, json, numpy as np, torch
from types import SimpleNamespace
from scipy.io import wavfile

WS = pathlib.Path.home() / "fs2"
sys.path.insert(0, str(WS / "FastSpeech2"))
from text import text_to_sequence
from utils.model import get_model, get_vocoder, vocoder_infer
dev = torch.device("cuda")

for name, lang in [("Svarah", "English"), ("TeluguMale", "Telugu")]:
    cfg = [yaml.safe_load(open(WS / f"FastSpeech2/config/{name}/{k}.yaml")) for k in ["preprocess", "model", "train"]]
    pp, mc, tc = cfg
    pdir = pathlib.Path(pp["path"]["preprocessed_path"])
    spk_map = json.load(open(pdir / "speakers.json"))
    model = get_model(SimpleNamespace(restore_step=25000), cfg, dev, train=False)
    voc = get_vocoder(mc, dev)
    hop, sr = pp["preprocessing"]["stft"]["hop_length"], pp["preprocessing"]["audio"]["sampling_rate"]
    out = WS / "diag" / lang
    out.mkdir(parents=True, exist_ok=True)

    T = lambda x, dt=torch.float: torch.from_numpy(np.asarray(x)).to(dt).unsqueeze(0).to(dev)
    vocode = lambda m, n: vocoder_infer(m[:, :n].transpose(1, 2), voc, mc, pp, lengths=[n * hop])[0]

    for line in open(pdir / "val.txt", encoding="utf-8").read().splitlines()[:2]:
        b, spk, ph, raw = line.split("|")
        ld = lambda k: np.load(pdir / k / f"{spk}-{k}-{b}.npy")
        mel, pit, eng, dur = ld("mel"), ld("pitch"), ld("energy"), ld("duration")
        seq = np.array(text_to_sequence(ph, pp["preprocessing"]["text"]["text_cleaners"]))
        s, txt = torch.tensor([spk_map[spk]], device=dev), T(seq, torch.long)
        L, n = torch.tensor([len(seq)], device=dev), mel.shape[0]

        shutil.copy(pathlib.Path(pp["path"]["raw_path"]) / spk / f"{b}.wav", out / f"{b}_0_raw.wav")
        with torch.no_grad():
            wavfile.write(out / f"{b}_1_gtmel.wav", sr, vocode(T(mel), n))
            o = model(s, txt, L, len(seq), mels=T(mel), mel_lens=torch.tensor([n], device=dev), max_mel_len=n,
                      p_targets=T(pit), e_targets=T(eng), d_targets=T(dur, torch.long))
            wavfile.write(out / f"{b}_2_fs2_teacher.wav", sr, vocode(o[1], n))
            o = model(s, txt, L, len(seq))
            wavfile.write(out / f"{b}_3_fs2_pred.wav", sr, vocode(o[1], o[9][0].item()))
        print(lang, b, "|", raw[:60])