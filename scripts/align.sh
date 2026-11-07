set -e
WS=~/fs2; MM=$WS/bin/micromamba; export MAMBA_ROOT_PREFIX=$WS/mamba MFA_ROOT_DIR=$WS/mfa_root
RAW=$WS/data/raw_data; PRE=$WS/data/preprocessed_data; LOG=$WS/logs; mkdir -p $LOG
MFA="$MM run -n mfa mfa"; PY="$MM run -n fs2 python"
t(){ echo "[$(date +%T)] $*"; }

t "1/6 downloading English MFA models"
for k in acoustic dictionary g2p; do $MFA model download $k english_us_arpa > $LOG/mfa_download.log 2>&1; done

t "2/6 English: finding words missing from the dictionary (OOV)"
$PY - << 'PY'
import re, glob, os
H = os.path.expanduser("~/fs2")
lex = {l.split("\t")[0] for l in open(f"{H}/mfa_root/pretrained_models/dictionary/english_us_arpa.dict")}
vocab = set()
for p in glob.glob(f"{H}/data/raw_data/Svarah/*/*.lab"):
    vocab.update(w.strip("'") for w in re.findall(r"[a-z']+", open(p).read()))
vocab.discard("")
oov = sorted(vocab - lex); open(f"{H}/logs/oov.txt", "w").write("\n".join(oov))
print(f"   vocab {len(vocab)} | OOV {len(oov)}")
PY

t "3/6 English: G2P for OOVs + merged lexicon"
$MFA g2p $LOG/oov.txt english_us_arpa $LOG/oov.dict --clean > $LOG/mfa_en_g2p.log 2>&1
$PY - << 'PY'
import re, glob, os
H = os.path.expanduser("~/fs2"); lex = {}
for path in [f"{H}/mfa_root/pretrained_models/dictionary/english_us_arpa.dict", f"{H}/logs/oov.dict"]:
    for line in open(path):
        f = line.rstrip("\n").split("\t")
        if len(f) >= 2: lex.setdefault(f[0], []).append(f[-1])
vocab = set()
for p in glob.glob(f"{H}/data/raw_data/Svarah/*/*.lab"):
    vocab.update(w.strip("'") for w in re.findall(r"[a-z']+", open(p).read()))
vocab.discard("")
with open(f"{H}/data/raw_data/svarah_lexicon.txt", "w") as out:
    for w in sorted(vocab):
        for pron in lex.get(w, []): out.write(f"{w}\t{pron}\n")
print("   still missing:", [w for w in vocab if w not in lex][:10])
PY

t "4/6 English: aligning (32 jobs)"
$MFA align $RAW/Svarah $RAW/svarah_lexicon.txt english_us_arpa $PRE/Svarah/TextGrid \
    -j 32 --clean --beam 100 --retry_beam 400 > $LOG/mfa_en_align.log 2>&1
t "   EN TextGrids: $(ls $PRE/Svarah/TextGrid/*/*.TextGrid | wc -l) / 3975"

t "5/6 Telugu: training aligner (1 speaker split into 16 chunks -> 16 jobs)"
C=$WS/mfa_corpus_te; rm -rf $C; mkdir -p $C
i=0; for w in $RAW/TeluguMale/te_male/*.wav; do
  k=$(printf "chunk%02d" $((i % 16))); mkdir -p $C/$k
  ln -s $w $C/$k/; ln -s ${w%.wav}.lab $C/$k/; i=$((i+1)); done
$MFA train $C $RAW/telugu_lexicon.txt $WS/data/telugu_mfa_acoustic.zip \
    --output_directory $WS/te_tg_tmp -j 16 --clean > $LOG/mfa_te_train.log 2>&1
mkdir -p $PRE/TeluguMale/TextGrid/te_male && mv $WS/te_tg_tmp/*/*.TextGrid $PRE/TeluguMale/TextGrid/te_male/
t "   TE TextGrids: $(ls $PRE/TeluguMale/TextGrid/te_male/*.TextGrid | wc -l) / 2809"

t "6/6 quality check"
$PY - << 'PY'
import tgt, glob, os
H = os.path.expanduser("~/fs2/data/preprocessed_data")
for ds in ["Svarah", "TeluguMale"]:
    tgs = sorted(glob.glob(f"{H}/{ds}/TextGrid/*/*.TextGrid")); spn = sil = tot = short = 0
    for p in tgs:
        for iv in tgt.io.read_textgrid(p).get_tier_by_name("phones")._objects:
            tot += 1; spn += iv.text == "spn"; sil += iv.text in ("", "sil", "sp")
            short += (iv.end_time - iv.start_time) * 22050 / 256 < 1
    print(f"   {ds}: {len(tgs)} files | spn {spn/tot:.2%} | silence {sil/tot:.2%} | <1 frame {short/tot:.2%}")
    tier = tgt.io.read_textgrid(tgs[0]).get_tier_by_name("phones")
    print("     ", [(iv.text, round(iv.end_time - iv.start_time, 3)) for iv in tier._objects[:10]])
PY
t "ALL DONE"
