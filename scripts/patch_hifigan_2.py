import pathlib
p = pathlib.Path.home() / "fs2/hifi-gan/train.py"
s = p.read_text()
for old, new in [("torch.backends.cudnn.benchmark = True", "torch.backends.cudnn.benchmark = False"),
                 ("if steps % a.validation_interval == 0:", "if steps % a.validation_interval == 0 and steps != 0:")]:
    print(("ok   " if old in s else "MISSING ") + old)
    s = s.replace(old, new)
p.write_text(s)