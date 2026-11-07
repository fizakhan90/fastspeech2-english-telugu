import re, pathlib

home = pathlib.Path.home()
root = home / "fs2/FastSpeech2"

# 1) Telugu phones -> text/symbols.py
phones = set()
for line in open(home / "fs2/data/raw_data/telugu_lexicon.txt", encoding="utf-8"):
    parts = line.strip().split()
    if len(parts) > 1:
        phones.update(parts[1:])
phones = sorted(phones)
print(len(phones), "Telugu phones:", " ".join(phones))

p = root / "text/symbols.py"
s = p.read_text()
if "_telugu" not in s:
    s = s.replace("_silences = ", f"_telugu = ['@' + x for x in {phones!r}]\n_silences = ", 1)
    s = re.sub(r"(\n\s*\+\s*_silences\s*\n?\s*\))", r"\n    + _telugu\1", s, count=1) \
        if re.search(r"\+\s*_silences\s*\n?\s*\)", s) else s.replace("+ _silences", "+ _telugu\n    + _silences", 1)
    # de-duplicate (some Telugu phones like @m already exist via pinyin)
    s += "\n_seen = set()\nsymbols = [x for x in symbols if not (x in _seen or _seen.add(x))]\n"
    p.write_text(s)
    print("symbols.py patched")

# 2) MFA3 empty intervals = silence
p = root / "preprocessor/preprocessor.py"
s = p.read_text()
s = s.replace("tgt.io.read_textgrid(tg_path)",
              "tgt.io.read_textgrid(tg_path, include_empty_intervals=True)")
s = s.replace("s, e, p = t.start_time, t.end_time, t.text",
              's, e, p = t.start_time, t.end_time, (t.text or "sp")')
p.write_text(s)
print("include_empty_intervals:", "include_empty_intervals=True" in s,
      "| empty->sp:", '(t.text or "sp")' in s)