import re, pathlib

root = pathlib.Path.home() / "fs2/FastSpeech2"

# 1) librosa >= 0.10: keyword-only args
for rel in ["audio/stft.py", "audio/audio_processing.py"]:
    p = root / rel
    s = p.read_text()
    s0 = s
    s = re.sub(
        r"librosa_mel_fn\(\s*sampling_rate,\s*filter_length,\s*n_mel_channels,\s*mel_fmin,\s*mel_fmax\s*\)",
        "librosa_mel_fn(sr=sampling_rate, n_fft=filter_length, n_mels=n_mel_channels, fmin=mel_fmin, fmax=mel_fmax)",
        s,
    )
    s = re.sub(r"pad_center\((\w+),\s*(\w+)\)", r"pad_center(\1, size=\2)", s)
    if s != s0:
        p.write_text(s)
        print("librosa patched:", rel)
    else:
        print("librosa unchanged (check!):", rel)

# 2) torch >= 2.6: add weights_only=False to every torch.load(...)
def patch_torch_load(s):
    out, i, n = [], 0, 0
    key = "torch.load("
    while True:
        j = s.find(key, i)
        if j < 0:
            out.append(s[i:])
            break
        k, depth = j + len(key), 1
        while depth:  # find the matching ')'
            depth += {"(": 1, ")": -1}.get(s[k], 0)
            k += 1
        call = s[j:k]
        if "weights_only" not in call:
            call = call[:-1].rstrip().rstrip(",") + ", weights_only=False)"
            n += 1
        out.append(s[i:j] + call)
        i = k
    return "".join(out), n

for p in root.rglob("*.py"):
    s, n = patch_torch_load(p.read_text())
    if n:
        p.write_text(s)
        print(f"torch.load x{n}:", p.relative_to(root))