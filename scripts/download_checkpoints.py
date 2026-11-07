"""Download the trained checkpoints from Hugging Face into English/ and Telugu/checkpoints/."""
import pathlib, shutil
from huggingface_hub import hf_hub_download
ROOT = pathlib.Path(__file__).resolve().parents[1]
for lang in ["English", "Telugu"]:
    for f in ["25000.pth.tar", "hifigan_g_00020000"]:
        p = hf_hub_download("fizakhan90/fastspeech2-english-telugu", f"{lang}/checkpoints/{f}")
        shutil.copy(p, ROOT / lang / "checkpoints" / f); print("ok", lang, f)
