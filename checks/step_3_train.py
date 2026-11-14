"""Train a tiny model on CPU and watch the loss fall"""

import importlib
import math

import numpy as np


def _data(fixtures):
    tok = importlib.import_module("tokeniser")
    text = (fixtures / "brewly_practice.txt").read_text(encoding="utf-8")
    try:
        vocab = tok.build_vocab(text)
        train, val = tok.split_data(tok.encode(text, vocab))
    except NotImplementedError:
        raise AssertionError("This step trains on text from your tokeniser. Finish step 1 first.") from None
    return vocab, np.asarray(train), np.asarray(val)


def test_untrained_model_loss_is_log_vocab(fixtures):
    """A table of zeros scores every character equally, so its loss is log(vocabulary size)"""
    bigram = importlib.import_module("bigram")
    vocab, train, val = _data(fixtures)
    got = bigram.loss(np.zeros((len(vocab), len(vocab))), val)
    assert math.isclose(got, math.log(len(vocab)), rel_tol=1e-6), f"Equal scores should give log({len(vocab)}) = {math.log(len(vocab)):.3f}; got {got:.3f}. Check you softmax each row and average -log of the right character's probability."


def test_loss_is_lower_for_a_better_table(fixtures):
    """loss rewards a table that puts its score on the true next character"""
    bigram = importlib.import_module("bigram")
    data = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    good = np.array([[0.0, 5.0], [5.0, 0.0]])
    bad = np.array([[5.0, 0.0], [0.0, 5.0]])
    assert bigram.loss(good, data) < 0.1 < 3 < bigram.loss(bad, data), "A table that always predicts the true next character should score near 0; one that always gets it wrong should score above 3. Check which entry of each row you take."


def test_training_beats_the_starting_loss_on_unseen_text(fixtures):
    """After 300 steps the validation loss is clearly below the starting loss"""
    bigram = importlib.import_module("bigram")
    vocab, train, val = _data(fixtures)
    start = math.log(len(vocab))
    logits = bigram.train(train, len(vocab), steps=300, seed=0)
    assert np.shape(logits) == (len(vocab), len(vocab)), f"train should return a table of shape ({len(vocab)}, {len(vocab)}): one row per input character."
    got = bigram.loss(logits, val)
    assert got < start - 0.5, f"Validation loss is {got:.2f} after training; the starting loss is {start:.2f}. It should fall by at least 0.5. Check the gradient is (probabilities - one-hot target) and you subtract lr times it."


def test_training_lowers_the_training_loss_too(fixtures):
    """The training loss also falls well below the starting loss"""
    bigram = importlib.import_module("bigram")
    vocab, train, _ = _data(fixtures)
    logits = bigram.train(train, len(vocab), steps=300, seed=0)
    got = bigram.loss(logits, train)
    assert got < math.log(len(vocab)) - 0.5, f"Training loss is {got:.2f} and should be at least 0.5 below log({len(vocab)}) = {math.log(len(vocab)):.2f}. Is the update going downhill (a minus sign)?"


def test_same_seed_same_result(fixtures):
    """Training twice with the same seed gives the same table"""
    bigram = importlib.import_module("bigram")
    vocab, train, _ = _data(fixtures)
    one = bigram.train(train, len(vocab), steps=20, seed=3)
    two = bigram.train(train, len(vocab), steps=20, seed=3)
    assert np.allclose(one, two), "Two runs with the same seed should match. Make one rng with np.random.default_rng(seed) and pass it to every get_batch call."
