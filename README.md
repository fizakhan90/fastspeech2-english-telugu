# FastSpeech 2 — English (Svarah) & Telugu (IndicTTS)

Two FastSpeech 2 acoustic models built on [ming024/FastSpeech2](https://github.com/ming024/FastSpeech2),
with HiFi-GAN vocoders fine-tuned on ground-truth-aligned (GTA) mels. Full technical details and
analysis: `Report/Technical_Report.pdf`.

## Repository layout
| Path | Contents |
|---|---|
| `English/`, `Telugu/` | `train.py`, `preprocess.py`, `inference.py` (thin wrappers), `config.yaml`, `checkpoints/` |
| `Results/<lang>/` | loss plots + CSVs, `test_generated_audios/`, `variance_adaptor_plots/{duration,pitch,energy,values}` |
| `Results/listening_test/` | blind listening-test ratings |
| `FastSpeech2/` | patched ming024 code shared by both languages |
| `hifi-gan/` | patched [jik876/hifi-gan](https://github.com/jik876/hifi-gan) used for vocoder fine-tuning |
| `frontend/` | English text normalizer, rule-based Telugu G2P, lexicons |
| `common.py` | shared helpers: config loading, data split, text frontends, synthesis |
| `scripts/` | alignment, data split, GTA generation, loss export, diagnostics, listening-test app |
| `data/splits/` | fixed train/val file lists (seed 42) |

Each `config.yaml` merges ming024's three files into `preprocess:`, `model:`, `train:` sections,
plus an `inference:` section (checkpoint steps, inference speaker).

## Setup
```bash
pip install -r requirements.txt            # Python 3.11, CUDA 12.x
python -c "import nltk; [nltk.download(p) for p in ['cmudict','averaged_perceptron_tagger_eng']]"
python scripts/download_checkpoints.py     # checkpoints from Hugging Face
```
Alignment additionally needs Montreal Forced Aligner 3.x in a separate conda env.

## Inference (uses the included checkpoints)
```bash
python English/inference.py                # 5 test sentences -> Results/English/
python Telugu/inference.py
python English/inference.py --text "Hello world." --id demo --out my_outputs
```

## Reproducing training
1. Data: Svarah (English, multi-speaker) and SPRINGLab/IndicTTS_Telugu (male speaker), resampled to
   22.05 kHz mono, in `data/raw_data/<Svarah|TeluguMale>/<speaker>/*.wav,*.lab`.
2. Alignment: `bash scripts/align.sh` (MFA English pretrained model; Telugu acoustic model trained with `mfa train`).
3. `python English/preprocess.py` / `python Telugu/preprocess.py`
4. `python English/train.py` / `python Telugu/train.py` (100k steps; step 25k selected by validation loss)
5. Vocoder: `scripts/make_gta.py` generates GTA mels, then fine-tune `hifi-gan/train.py --fine_tuning True`
   starting from the universal HiFi-GAN generator (step 20k selected by validation mel error).

## Data summary
| | Speakers | Train (raw / trimmed) | Val (raw / trimmed) |
|---|---|---|---|
| English (Svarah) | 115 | 5.12 h / 4.77 h | 0.97 h / 0.90 h |
| Telugu (IndicTTS, male) | 1 | 5.00 h / 4.27 h | 1.00 h / 0.85 h |

## Results summary
- FastSpeech 2: validation loss plateaus after ~15–25k steps while training loss keeps falling
  (overfitting on ~5 h); the 25k checkpoint is used.
- A stage-wise diagnostic (real audio → vocoded real mel → teacher-forced FS2 → free-running FS2)
  located the main quality loss in the acoustic model's over-smoothed mels, not the vocoder.
- GTA fine-tuning of HiFi-GAN targets that gap. Blind listening test: the fine-tuned vocoder was preferred in 13 of 15 judgments (2 raters).

## Changes to ming024/FastSpeech2
| File | Change |
|---|---|
| `audio/stft.py`, `audio/audio_processing.py` | librosa ≥0.10 keyword arguments (`filters.mel`, `pad_center`) |
| `utils/model.py` | `torch.load(..., weights_only=False)` for torch ≥2.6 |
| `utils/tools.py` | cast pitch/energy targets to float32 (dtype crash in backward) |
| `text/symbols.py` | added 46 Telugu phones (`@`-prefixed), de-duplicated symbol list |
| `preprocessor/preprocessor.py` | read MFA 3 empty intervals as silence (`sp`) |
| `config/Svarah/`, `config/TeluguMale/` | new configs (`max_seq_len` 1500, phoneme-level pitch/energy, 100k steps, anneal 60k/80k/90k) |

## Changes to jik876/hifi-gan
| File | Change |
|---|---|
| `meldataset.py` | librosa keyword arguments; `torch.stft(return_complex=True)` |
| `utils.py` | `torch.load(..., weights_only=False)` |
| `train.py` | fine-tune from a generator-only checkpoint; `cudnn.benchmark=False` and no step-0 validation (variable-length validation stalled the GPU) |

## New code
`common.py`, `English/*`, `Telugu/*`, `frontend/*` (English normalizer, Telugu G2P and number-to-words),
`scripts/*`.

## Checkpoints
Hosted on Hugging Face (https://huggingface.co/fizakhan90/fastspeech2-english-telugu) and downloaded into `<lang>/checkpoints/` by `python scripts/download_checkpoints.py`: FastSpeech 2 step 25000 (optimizer state removed),
fine-tuned HiFi-GAN generator step 20000, `stats.json`, `speakers.json`.

## Known limitations
Svarah is ASR data (16 kHz, background noise) → upsampled English lacks content above 8 kHz.
Context-free G2P: homographs ("conduct", "పట్టు") get one pronunciation. Indian names and "andaroo"
use US-English G2P. Spaces inside a Telugu word ("అడ విలో") do not change the phone sequence.
