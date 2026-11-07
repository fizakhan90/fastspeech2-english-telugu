import pathlib
H = pathlib.Path.home() / "fs2/hifi-gan"

def sub(rel, old, new):
    p = H / rel; s = p.read_text()
    if new in s: print("already:", rel); return
    assert old in s, f"NOT FOUND in {rel}: {old[:60]}"
    p.write_text(s.replace(old, new)); print("patched:", rel)

sub("meldataset.py", "librosa_mel_fn(sampling_rate, n_fft, num_mels, fmin, fmax)",
    "librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)")
sub("meldataset.py", "normalized=False, onesided=True)",
    "normalized=False, onesided=True, return_complex=True)")
sub("meldataset.py", "spec = torch.sqrt(spec.pow(2).sum(-1)+(1e-9))",
    "spec = torch.sqrt(spec.abs().pow(2)+(1e-9))")
sub("utils.py", "torch.load(filepath, map_location=device)",
    "torch.load(filepath, map_location=device, weights_only=False)")
sub("train.py",
    "        state_dict_do = None\n        last_epoch = -1\n",
    "        state_dict_do = None\n        last_epoch = -1\n"
    "        if cp_g is not None:  # fine-tune from generator-only (universal) checkpoint\n"
    "            generator.load_state_dict(load_checkpoint(cp_g, device)['generator'])\n"
    "            print('Loaded generator only from', cp_g)\n")