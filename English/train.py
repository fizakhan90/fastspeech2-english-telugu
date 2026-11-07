"""Train FastSpeech 2 (English). Usage: python English/train.py [--restore_step N]"""
import sys, pathlib, argparse
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common import load_config
configs, _ = load_config("English")
from train import main                     # ming024 FastSpeech2/train.py
ap = argparse.ArgumentParser(); ap.add_argument("--restore_step", type=int, default=0)
main(ap.parse_args(), configs)
