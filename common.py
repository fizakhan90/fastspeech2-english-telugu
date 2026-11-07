"""Shared helpers used by English/ and Telugu/ wrappers (paths, config, split, text frontends, synthesis)."""
import os, re, sys, json, pathlib, yaml
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parent
FS = ROOT / "FastSpeech2"
sys.path[:0] = [str(FS), str(ROOT / "frontend")]

def load_config(lang):
    """Read <lang>/config.yaml, make paths absolute, return ((preprocess, model, train), inference)."""
    cfg = yaml.safe_load(open(ROOT / lang / "config.yaml", encoding="utf-8"))
    pp, mc, tc = cfg["preprocess"], cfg["model"], cfg["train"]
    for sec in (pp["path"], tc["path"]):
        for k, v in sec.items():
            sec[k] = str(ROOT / v)
    os.chdir(FS)                         # ming024 uses paths relative to its own folder
    return (pp, mc, tc), cfg["inference"]

def apply_split(pp, prefix):
    """Replace ming024's random split with our fixed CSV split (data/splits)."""
    import pandas as pd
    pdir = pathlib.Path(pp["path"]["preprocessed_path"])
    lines = {l.split("|")[0]: l for l in open(pdir / "train.txt", encoding="utf-8").read().splitlines() if l}
    ids = {s: [pathlib.Path(p).stem for p in pd.read_csv(ROOT / "data/splits" / f"{prefix}_{s}.csv")["path"]]
           for s in ["train", "val"]}
    assert not set(ids["train"]) & set(ids["val"]), "train/val overlap"
    for s, L in ids.items():
        kept = [lines[i] for i in L if i in lines]
        (pdir / f"{s}.txt").write_text("\n".join(kept) + "\n", encoding="utf-8")
        print(f"{s}: {len(kept)} utterances ({len(L) - len(kept)} dropped: no alignment/pitch)")

def test_sentences(prefix):
    rows = [l.rstrip("\n").split("|", 1) for l in open(ROOT / "scripts/test_sentences.txt", encoding="utf-8") if "|" in l]
    return [(i, t) for i, t in rows if i.startswith(prefix)]

# ---------------- English frontend ----------------
def read_lex(p):
    lex = {}
    for l in open(p, encoding="utf-8"):
        t = l.strip().split()
        if len(t) > 1 and t[0].lower() not in lex:      # first pronunciation wins (as in ming024)
            lex[t[0].lower()] = t[1:]
    return lex

def en_phones(text, lex):
    from g2p_en import G2p
    from en_normalize import normalize_en
    g2p, ph = G2p(), []
    toks = [w.strip().lower() for w in re.split(r"([,;.\-\?\!\s+])", normalize_en(text)) if w.strip()]
    for k, w in enumerate(toks):
        nxt = toks[k + 1] if k + 1 < len(toks) else ""
        if w in ",;.-?!": ph.append("sp")
        elif w == "a" and len(nxt) == 1 and nxt.isalpha(): ph.append("EY1")   # spelled letter: "A M"
        elif w in lex: ph += lex[w]
        else: ph += [p for p in g2p(w) if p.strip() and p not in ",.?!;"]
    return ph

# ---------------- Telugu frontend ----------------
TE = ["సున్నా","ఒకటి","రెండు","మూడు","నాలుగు","ఐదు","ఆరు","ఏడు","ఎనిమిది","తొమ్మిది",
      "పది","పదకొండు","పన్నెండు","పదమూడు","పద్నాలుగు","పదిహేను","పదహారు","పదిహేడు","పద్దెనిమిది","పందొమ్మిది"]
TENS = {2:"ఇరవై",3:"ముప్పై",4:"నలభై",5:"యాభై",6:"అరవై",7:"డెబ్బై",8:"ఎనభై",9:"తొంభై"}

def te_num(n):
    if n < 20: return TE[n]
    if n < 100: return TENS[n // 10] + ("" if n % 10 == 0 else " " + TE[n % 10])
    if n < 1000:
        h, r = divmod(n, 100)
        if r == 0: return "వంద" if h == 1 else TE[h] + " వందలు"
        return ("నూట" if h == 1 else TE[h] + " వందల") + " " + te_num(r)
    if n < 100000:
        k, r = divmod(n, 1000)
        if r == 0: return "వెయ్యి" if k == 1 else te_num(k) + " వేలు"
        return ("వెయ్యి" if k == 1 else te_num(k) + " వేల") + " " + te_num(r)
    return " ".join(TE[int(c)] for c in str(n))

def te_ord(n):
    w = te_num(n)
    if w.endswith("ై"): return w[:-1] + "య్యవ"
    if w[-1] in "ుి": return w[:-1] + "వ"
    return w + "వ"

def te_phones(text):
    from te_g2p import clean, g2p_word
    t = clean(text)
    t = re.sub(r"(\d{1,2}):(\d{2})", lambda m: te_num(int(m[1])) + " " + te_num(int(m[2])), t)
    t = re.sub(r"(\d+)\s*వ(?![ఀ-౿])", lambda m: te_ord(int(m[1])), t)
    t = re.sub(r"\d+", lambda m: te_num(int(m[0])), t)
    ph = []
    for tok in re.findall(r"[ఀ-౿]+|[,.;:!?।|]", t):
        ph += ["sp"] if not re.match(r"[ఀ-౿]", tok) else g2p_word(tok)
    return ph

# ---------------- synthesis ----------------
def _plot(vals, phones, ylabel, title, path):
    fig, ax = plt.subplots(figsize=(max(8, .32 * len(phones)), 3.6))
    ax.bar(range(len(vals)), vals) if "uration" in ylabel else ax.plot(range(len(vals)), vals, "o-")
    ax.set_xticks(range(len(phones))); ax.set_xticklabels(phones, rotation=90, fontsize=8)
    ax.set_ylabel(ylabel); ax.set_title(title); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)

def synthesize(lang, sentences, out=None):
    import torch, pandas as pd, hifigan
    from types import SimpleNamespace
    from scipy.io import wavfile
    from text import text_to_sequence
    from utils.model import get_model, vocoder_infer

    (pp, mc, tc), inf = load_config(lang)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = ROOT / lang / "checkpoints"
    pp["path"]["preprocessed_path"] = str(ck)   # stats.json + speakers.json live here for inference
    stats = json.load(open(ck / "stats.json"))
    spk_id = json.load(open(ck / "speakers.json"))[inf["speaker"]]
    model = get_model(SimpleNamespace(restore_step=inf["fs2_step"]), (pp, mc, tc), dev, train=False)
    voc = hifigan.Generator(hifigan.AttrDict(json.load(open(FS / "hifigan/config.json"))))
    voc.load_state_dict(torch.load(ck / f"hifigan_g_{inf['vocoder_step']:08d}", map_location="cpu",
                                   weights_only=False)["generator"])
    voc.eval(); voc.remove_weight_norm(); voc = voc.to(dev)

    hop, sr = pp["preprocessing"]["stft"]["hop_length"], pp["preprocessing"]["audio"]["sampling_rate"]
    res = pathlib.Path(out) if out else ROOT / "Results" / lang
    dirs = {"audio": res / "test_generated_audios"}
    dirs.update({d: res / "variance_adaptor_plots" / d for d in ["duration", "pitch", "energy", "values"]})
    for d in dirs.values(): d.mkdir(parents=True, exist_ok=True)
    lex = read_lex(pp["path"]["lexicon_path"]) if lang == "English" else None

    for sid, text in sentences:
        phones = en_phones(text, lex) if lang == "English" else te_phones(text)
        while phones and phones[-1] == "sp": phones.pop()
        seq = np.array(text_to_sequence("{" + " ".join(phones) + "}", pp["preprocessing"]["text"]["text_cleaners"]))
        assert len(seq) == len(phones), f"{sid}: unknown phone(s) dropped"
        with torch.no_grad():
            o = model(speakers=torch.tensor([spk_id], device=dev),
                      texts=torch.from_numpy(seq).long().unsqueeze(0).to(dev),
                      src_lens=torch.tensor([len(seq)], device=dev), max_src_len=len(seq))
            n = o[9][0].item()
            wav = vocoder_infer(o[1][:, :n].transpose(1, 2), voc, mc, pp, lengths=[n * hop])[0]
        wavfile.write(dirs["audio"] / f"{sid}.wav", sr, wav)
        dur = o[5][0].cpu().numpy()
        pit = o[2][0].cpu().numpy() * stats["pitch"][3] + stats["pitch"][2]
        eng = o[3][0].cpu().numpy() * stats["energy"][3] + stats["energy"][2]
        pd.DataFrame({"phone": phones, "dur_frames": dur, "dur_ms": dur * hop / sr * 1000,
                      "pitch_hz": pit, "energy": eng}).to_csv(dirs["values"] / f"{sid}.csv", index=False)
        _plot(dur, phones, "Duration (frames)", f"{lang} {sid} – predicted duration", dirs["duration"] / f"{sid}.png")
        _plot(pit, phones, "F0 (Hz)", f"{lang} {sid} – predicted pitch", dirs["pitch"] / f"{sid}.png")
        _plot(eng, phones, "Energy", f"{lang} {sid} – predicted energy", dirs["energy"] / f"{sid}.png")
        print(f"{sid}: {len(phones)} phones -> {n * hop / sr:.2f} s | {' '.join(phones)}")
