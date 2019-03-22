import torch
import re
from collections import namedtuple
import numpy as np


UNK_TOKEN = "<unk>"
PAD_TOKEN = "<pad>"
INIT_TOKEN = "@@NULL@@"

SHIFT = 0
REDUCE = 1
SR_PAD = 2


def get_device(cfg):
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def filereader(path):
    """read SST lines"""
    with open(path, mode="r", encoding="utf-8") as f:
        for line in f:
            yield line.strip().replace("\\", "")


def tokens_from_treestring(s):
    """extract the tokens from a sentiment tree"""
    return re.findall(r"\([0-9] ([^\(\)]+)\)", s)


def token_labels_from_treestring(s):
    """extract token labels from sentiment tree"""
    return list(map(int, re.findall(r"\(([0-9]) [^\(\)]", s)))


def transitions_from_string(s):
  s = re.sub("\([0-5] ([^)]+)\)", "0", s)
  s = re.sub("\)", " )", s)
  s = re.sub("\([0-4] ", "", s)
  s = re.sub("\([0-4] ", "", s)
  s = re.sub("\)", "1", s)
  return list(map(int, s.split()))


def get_spans(s):
    """extract all spans given a treestring"""
    open_idx = []

    for i in range(len(s)):
        c = s[i]
        if c == "(":
            open_idx.append(i)
        elif c == ")":
            yield s[open_idx.pop():i + 1]


Example = namedtuple("Example",
                     ["tokens", "label", "transitions", "token_labels"])


def examplereader(path, lower=False, subphrases=False, min_length=1):
    """
    Reads in examples
    :param path:
    :param lower:
    :param subphrases: extract all subphrases
    :param min_length: minimum length for used phrases
    :return:
    """
    for line in filereader(path):
        line = line.lower() if lower else line
        line = re.sub("\\\\", "", line)  # fix escape

        if subphrases:
            phrases = get_spans(line)
        else:
            phrases = [line]

        for phrase in phrases:
            tokens = tokens_from_treestring(phrase)

            # skip short phrases
            if len(tokens) < min_length:
                continue

            label = int(phrase[1])
            trans = transitions_from_string(phrase)
            token_labels = token_labels_from_treestring(phrase)
            assert len(tokens) == len(token_labels), "mismatch tokens/labels"
            yield Example(tokens=tokens, label=label,
                          transitions=trans, token_labels=token_labels)
            
            
def load_glove(glove_path, vocab, glove_dim=300):
    """
    Load Glove embeddings and update vocab.
    :param glove_path:
    :param vocab:
    :param glove_dim:
    :return:
    """
    vectors = []
    w2i = {}
    i2w = []

    # Random embedding vector for unknown words
    vectors.append(np.random.uniform(
        -0.05, 0.05, glove_dim).astype(np.float32))
    w2i[UNK_TOKEN] = 0
    i2w.append(UNK_TOKEN)

    # Zero vector for padding
    vectors.append(np.zeros(glove_dim).astype(np.float32))
    w2i[PAD_TOKEN] = 1
    i2w.append(PAD_TOKEN)

    with open(glove_path, mode="r", encoding="utf-8") as f:
        for line in f:
            word, vec = line.split(u' ', 1)
            w2i[word] = len(vectors)
            i2w.append(word)
            vectors.append(np.array(vec.split(), dtype=np.float32))

    # fix brackets
    w2i[u'-LRB-'] = w2i.pop(u'(')
    w2i[u'-RRB-'] = w2i.pop(u')')

    i2w[w2i[u'-LRB-']] = u'-LRB-'
    i2w[w2i[u'-RRB-']] = u'-RRB-'

    vocab.w2i = w2i
    vocab.i2w = i2w

    return np.stack(vectors)


from collections import Counter, OrderedDict, defaultdict


class OrderedCounter(Counter, OrderedDict):
    """Counter that remembers the order elements are first seen"""

    def __repr__(self):
        return '%s(%r)' % (self.__class__.__name__,
                           OrderedDict(self))

    def __reduce__(self):
        return self.__class__, (OrderedDict(self),)


class Vocabulary:
    """A vocabulary, assigns IDs to tokens"""

    def __init__(self):
        self.freqs = OrderedCounter()
        self.w2i = {}
        self.i2w = []

    def count_token(self, t):
        self.freqs[t] += 1

    def add_token(self, t):
        self.w2i[t] = len(self.w2i)
        self.i2w.append(t)

    def build(self, min_freq=0):
        self.add_token(UNK_TOKEN)  # reserve 0 for <unk>
        self.add_token(PAD_TOKEN)  # reserve 1 for <pad> (discussed later)

        tok_freq = list(self.freqs.items())
        tok_freq.sort(key=lambda x: x[1], reverse=True)
        for tok, freq in tok_freq:
            if freq >= min_freq:
                self.add_token(tok)