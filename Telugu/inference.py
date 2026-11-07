"""Synthesize Telugu speech + variance-adaptor plots.
Usage: python Telugu/inference.py                       # the 5 test sentences -> Results/Telugu/
       python Telugu/inference.py --text "..." --id my1 --out some_dir"""
import sys, pathlib, argparse
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
ap = argparse.ArgumentParser()
ap.add_argument("--text"); ap.add_argument("--id", default="custom"); ap.add_argument("--out")
a = ap.parse_args()
out = pathlib.Path(a.out).resolve() if a.out else None
from common import synthesize, test_sentences
synthesize("Telugu", [(a.id, a.text)] if a.text else test_sentences("T"), out)
