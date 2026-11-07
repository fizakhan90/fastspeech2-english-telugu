import re, unicodedata
VIRAMA, ANUS, VIS, CANDRA = "్", "ం", "ః", "ఁ"
V = {"అ":"a","ఆ":"aa","ఇ":"i","ఈ":"ii","ఉ":"u","ఊ":"uu","ఋ":"r u","ౠ":"r uu",
     "ఎ":"e","ఏ":"ee","ఐ":"ai","ఒ":"o","ఓ":"oo","ఔ":"au"}
M = {"ా":"aa","ి":"i","ీ":"ii","ు":"u","ూ":"uu","ృ":"r u","ౄ":"r uu",
     "ె":"e","ే":"ee","ై":"ai","ొ":"o","ో":"oo","ౌ":"au"}
C = {"క":"k","ఖ":"kh","గ":"g","ఘ":"gh","ఙ":"ng","చ":"c","ఛ":"ch","జ":"j","ఝ":"jh","ఞ":"ny",
     "ట":"tt","ఠ":"tth","డ":"dd","ఢ":"ddh","ణ":"nn","త":"t","థ":"th","ద":"d","ధ":"dh","న":"n",
     "ప":"p","ఫ":"ph","బ":"b","భ":"bh","మ":"m","య":"y","ర":"r","ఱ":"r","ల":"l","ళ":"ll",
     "వ":"v","శ":"sh","ష":"ss","స":"s","హ":"h"}
NASAL = {**dict.fromkeys(["k","kh","g","gh"], "ng"), **dict.fromkeys(["c","ch","j","jh"], "ny"),
         **dict.fromkeys(["tt","tth","dd","ddh"], "nn"), **dict.fromkeys(["t","th","d","dh","n"], "n"),
         **dict.fromkeys(["p","ph","b","bh","m"], "m")}

def clean(text):
    text = unicodedata.normalize("NFC", text).replace("‌", "").replace("‍", "")
    return re.sub(r"(?<=[ఀ-౿])[oO0]", ANUS, text)   # Latin o / zero typed for anusvara

def words(text):
    return re.findall(r"[ఀ-౿]+", clean(text))

def g2p_word(w):
    out, i = [], 0
    while i < len(w):
        ch, nxt = w[i], (w[i+1] if i + 1 < len(w) else "")
        if ch in C:
            out += C[ch].split()
            if nxt == VIRAMA: i += 2; continue
            if nxt in M:      out += M[nxt].split(); i += 2; continue
            out.append("a")
        elif ch in V:    out += V[ch].split()
        elif ch == ANUS: out.append("<N>")
        elif ch == VIS:  out.append("h")
        i += 1
    return [NASAL.get(out[j+1] if j + 1 < len(out) else None, "m") if p == "<N>" else p
            for j, p in enumerate(out)]
