"""Preprocess English: mel/pitch/energy/duration from MFA TextGrids, then apply the fixed split.
Run scripts/align.sh first. Usage: python English/preprocess.py"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common import load_config, apply_split
configs, _ = load_config("English")
from preprocessor.preprocessor import Preprocessor
Preprocessor(configs[0]).build_from_path()
apply_split(configs[0], "en")
