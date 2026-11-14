"""Build a character-level tokeniser"""

import importlib

TEXT = "It is a truth universally acknowledged.\nBrewly: 120 cafes!"


def test_vocab_is_the_sorted_characters():
    """The vocabulary is the sorted list of different characters"""
    tok = importlib.import_module("tokeniser")
    vocab = tok.build_vocab("banana")
    assert list(vocab) == ["a", "b", "n"], f"For 'banana' the vocabulary should be ['a', 'b', 'n']; got {list(vocab)!r}. Each character appears once, in order."


def test_encode_gives_one_id_per_character():
    """encode gives one whole-number id for each character"""
    tok = importlib.import_module("tokeniser")
    vocab = tok.build_vocab("banana")
    ids = list(tok.encode("banana", vocab))
    assert ids == [1, 0, 2, 0, 2, 0], f"'banana' should encode to [1, 0, 2, 0, 2, 0], the position of each letter in the vocabulary; got {ids}."


def test_decode_reverses_encode():
    """decode(encode(text)) gives back the same text, including spaces, new lines and punctuation"""
    tok = importlib.import_module("tokeniser")
    vocab = tok.build_vocab(TEXT)
    back = tok.decode(tok.encode(TEXT, vocab), vocab)
    assert back == TEXT, f"Decoding the encoded text should give the text back exactly; got {back!r}. Check decode joins with an empty string."


def test_split_is_ninety_ten_without_shuffling():
    """split_data keeps the first 90% for training and the last 10% for validation"""
    tok = importlib.import_module("tokeniser")
    train, val = tok.split_data(list(range(100)))
    assert (len(train), len(val)) == (90, 10), f"100 ids should split into 90 and 10; got {len(train)} and {len(val)}."
    assert list(train) == list(range(90)) and list(val) == list(range(90, 100)), "Do not shuffle: train is the first 90 ids and validation is the last 10. Text has an order, and shuffling would leak the end of the book into training."
