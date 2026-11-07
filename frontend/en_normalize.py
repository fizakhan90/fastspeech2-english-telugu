import re, inflect
from unidecode import unidecode
_p = inflect.engine()
_MONTHS = r"(january|february|march|april|may|june|july|august|september|october|november|december)"
_DIG = "zero one two three four five six seven eight nine".split()

def _num(n):  return _p.number_to_words(int(n), andword="").replace(",", "").replace("-", " ")
def _ord(n):  return _p.ordinal(_num(n)).replace("-", " ")
def _two(n):  n = int(n); return "oh " + _num(n) if n < 10 else _num(n)
def _digits(s): return " ".join(_DIG[int(c)] for c in s)
def _dec(s):
    a, _, b = s.partition(".")
    return _num(a) + (" point " + _digits(b) if b else "")

def _year(y):
    y = int(y)
    if 2000 <= y <= 2009 or y % 1000 == 0: return _num(y)
    hi, lo = divmod(y, 100)
    return _num(hi) + (" hundred" if lo == 0 else " " + _two(lo))

def _time(m):
    h, mm, ap = m.group(1), m.group(2), m.group(3)
    s = _num(h) + ("" if int(mm) == 0 else " " + _two(mm))
    return s + (" " + " ".join(ap.lower().replace(".", "")) if ap else "")

def normalize_en(t):
    t = unidecode(t)
    t = re.sub(r"\b(\d{1,2}):(\d{2})\s*([AaPp]\.?[Mm]\.?)?", _time, t)
    t = re.sub(r"\b(\d{1,2})(?:st|nd|rd|th)?\s+" + _MONTHS,
               lambda m: _ord(m.group(1)) + " " + m.group(2), t, flags=re.I)
    t = re.sub(r"\b(\d+)(st|nd|rd|th)\b", lambda m: _ord(m.group(1)), t)
    t = re.sub(r"\d{5,}", lambda m: _digits(m.group(0)), t)
    t = re.sub(r"\b\d{1,3}(?:,\d{3})+\b", lambda m: _num(m.group(0).replace(",", "")), t)
    t = re.sub(r"(\d+(?:\.\d+)?)\s*%", lambda m: _dec(m.group(1)) + " percent", t)
    t = re.sub(r"\d+\.\d+", lambda m: _dec(m.group(0)), t)
    t = re.sub(r"\b(1[1-9]\d\d|20\d\d)\b", lambda m: _year(m.group(1)), t)
    t = re.sub(r"\d+", lambda m: _num(m.group(0)), t)
    t = t.lower().replace("&", " and ")
    t = re.sub(r"[^a-z' ,.?!]", " ", t)
    return re.sub(r"\s+", " ", t).strip()
