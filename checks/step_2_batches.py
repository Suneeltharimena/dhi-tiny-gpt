"""Make batches and know the starting loss"""

import importlib
import math

import numpy as np


def test_batch_shapes():
    """get_batch returns inputs and targets of shape (batch_size, block_size)"""
    b = importlib.import_module("batches")
    x, y = b.get_batch(list(range(50)), 8, 4, np.random.default_rng(0))
    assert np.shape(x) == (4, 8) and np.shape(y) == (4, 8), f"Both x and y should have shape (4, 8) for batch_size 4 and block_size 8; got {np.shape(x)} and {np.shape(y)}."


def test_targets_are_inputs_moved_one_place():
    """Each target is the character that follows its input"""
    b = importlib.import_module("batches")
    data = list(range(100, 200))  # the value tells us the position, so shifting is easy to see
    x, y = b.get_batch(data, 8, 16, np.random.default_rng(1))
    assert (np.asarray(y) == np.asarray(x) + 1).all(), "Every target should be its input plus 1 on this counting data: y is x moved one place along, so the model learns to predict the next character."
    assert (np.asarray(x) >= 100).all() and (np.asarray(y) <= 199).all(), "A window ran off the end of the data. The last start position must leave room for block_size + 1 characters."


def test_batch_uses_the_generator_it_is_given():
    """The same seed gives the same batch, and a different seed gives a different one"""
    b = importlib.import_module("batches")
    data = list(range(1000))
    first = b.get_batch(data, 8, 4, np.random.default_rng(5))[0]
    again = b.get_batch(data, 8, 4, np.random.default_rng(5))[0]
    other = b.get_batch(data, 8, 4, np.random.default_rng(6))[0]
    assert (np.asarray(first) == np.asarray(again)).all(), "The same seed should give the same batch. Take your random numbers only from the rng you are given."
    assert not (np.asarray(first) == np.asarray(other)).all(), "A different seed should give a different batch. Choose the start positions at random with rng.integers."


def test_starting_loss_is_log_of_vocab_size():
    """An untrained model's loss is log(vocabulary size): about 4.17 for 65 characters"""
    b = importlib.import_module("batches")
    got = b.starting_loss(65)
    assert math.isclose(got, math.log(65)), f"With 65 characters the starting loss is log(65), about 4.17; got {got}. Guessing evenly gives each character a 1 in 65 chance."
