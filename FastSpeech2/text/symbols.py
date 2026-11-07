""" from https://github.com/keithito/tacotron """

"""
Defines the set of symbols used in text input to the model.

The default is a set of ASCII characters that works well for English or text that has been run through Unidecode. For other data, you can modify _characters. See TRAINING_DATA.md for details. """

from text import cmudict, pinyin

_pad = "_"
_punctuation = "!'(),.:;? "
_special = "-"
_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
_telugu = ['@' + x for x in ['a', 'aa', 'ai', 'au', 'b', 'bh', 'c', 'ch', 'd', 'dd', 'ddh', 'dh', 'e', 'ee', 'g', 'gh', 'h', 'i', 'ii', 'j', 'jh', 'k', 'kh', 'l', 'll', 'm', 'n', 'ng', 'nn', 'ny', 'o', 'oo', 'p', 'ph', 'r', 's', 'sh', 'ss', 't', 'th', 'tt', 'tth', 'u', 'uu', 'v', 'y']]
_silences = ["@sp", "@spn", "@sil"]

# Prepend "@" to ARPAbet symbols to ensure uniqueness (some are the same as uppercase letters):
_arpabet = ["@" + s for s in cmudict.valid_symbols]
_pinyin = ["@" + s for s in pinyin.valid_symbols]

# Export all symbols:
symbols = (
    [_pad]
    + list(_special)
    + list(_punctuation)
    + list(_letters)
    + _arpabet
    + _pinyin
    + _telugu
    + _silences
)

_seen = set()
symbols = [x for x in symbols if not (x in _seen or _seen.add(x))]
