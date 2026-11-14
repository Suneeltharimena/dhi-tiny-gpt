"""Step 1: a character-level tokeniser.

A character-level GPT reads one character at a time, so its vocabulary is just the
characters in the book. Build the three pieces nanoGPT's prepare.py builds.
"""


def build_vocab(text):
    """Return the sorted list of the different characters in text. A character's position in the list is its id."""
    raise NotImplementedError("Step 1: return sorted(set(text)) as a list")


def encode(text, vocab):
    """Turn text into a list of ids, one per character."""
    raise NotImplementedError("Step 1: look up each character's position in vocab")


def decode(ids, vocab):
    """Turn a list of ids back into text. decode(encode(t, v), v) must give back t."""
    raise NotImplementedError("Step 1: join the character for each id")


def split_data(ids, train_fraction=0.9):
    """Return (train, validation). The first 90% trains the model; the last 10% is kept to test it on unseen text."""
    raise NotImplementedError("Step 1: cut the list at int(len(ids) * train_fraction); do not shuffle")
