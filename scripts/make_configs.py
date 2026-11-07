import yaml, pathlib, copy

home = pathlib.Path.home()
WS = home / "fs2"
FS = WS / "FastSpeech2"
src = FS / "config/LJSpeech"
base = {k: yaml.safe_load(open(src / f"{k}.yaml")) for k in ["preprocess", "model", "train"]}

langs = {
    "Svarah":     dict(lang="en", cleaners=["english_cleaners"], multi=True,
                       lexicon=WS / "data/raw_data/svarah_lexicon.txt"),
    "TeluguMale": dict(lang="te", cleaners=[], multi=False,
                       lexicon=WS / "data/raw_data/telugu_lexicon.txt"),
}

for name, c in langs.items():
    pp, md, tr = (copy.deepcopy(base[k]) for k in ["preprocess", "model", "train"])

    # preprocess.yaml
    pp["dataset"] = name
    pp["path"].update(
        corpus_path=str(WS / "data/raw_data" / name),   # unused (prepare_align skipped)
        lexicon_path=str(c["lexicon"]),
        raw_path=str(WS / "data/raw_data" / name),
        preprocessed_path=str(WS / "data/preprocessed_data" / name),
    )
    pp["preprocessing"]["val_size"] = 0                 # we use our own CSV split
    pp["preprocessing"]["text"].update(text_cleaners=c["cleaners"], language=c["lang"])
    pp["preprocessing"]["pitch"]["feature"] = "phoneme_level"
    pp["preprocessing"]["energy"]["feature"] = "phoneme_level"

    # model.yaml
    md["multi_speaker"] = c["multi"]
    md["max_seq_len"] = 1500
    md["vocoder"]["speaker"] = "universal"

    # train.yaml
    tr["path"].update(
        ckpt_path=str(WS / "output/ckpt" / name),
        log_path=str(WS / "output/log" / name),
        result_path=str(WS / "output/result" / name),
    )
    tr["optimizer"]["batch_size"] = 16
    tr["optimizer"]["anneal_steps"] = [60000, 80000, 90000]  # default 300k+ never triggers in 100k
    tr["step"].update(total_step=100000, log_step=100, val_step=1000,
                      synth_step=5000, save_step=5000)

    out = FS / "config" / name
    out.mkdir(parents=True, exist_ok=True)
    for k, d in zip(["preprocess", "model", "train"], [pp, md, tr]):
        yaml.safe_dump(d, open(out / f"{k}.yaml", "w"), sort_keys=False, allow_unicode=True)
    print("wrote", out)