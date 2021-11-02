import re

punkt = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~ '


def detokenize(text):
    text = text.replace(" .", ".").replace(" ,", ",").replace(" !", "!")
    text = text.replace(" ?", "?").replace(" )", ")").replace("( ", "(")
    return text


def drop_bad_words(text, max_len=30, return_digits=None):
    parts = re.split('(\\W)', text)
    if max_len:
        parts = [w for w in parts if len(w) <= max_len]
    if return_digits is not None:
        parts = [str(return_digits) if p == 'DIGIT' else p for p in parts]
    return ''.join(parts)


def text_preprocess(text):
    # strip punctuation on the left
    text = text.lstrip(punkt)
    # remove exrea spaces after tokenization
    text = detokenize(text)
    # remove too long words because generally they confuse a seq2seq model
    # and often they are meaningless combinations of characters
    text = drop_bad_words(text)
    return text


def text_postprocess(text):
    # strip multiple punctuation on the rigth
    res2 = text.rstrip(punkt)
    if len(res2) < len(text):
        res2 += text[len(res2)]
    return res2
