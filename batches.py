"""Step 2: batches and the starting loss.

A GPT learns to predict the next character. A batch is a stack of short windows of the
text (the inputs, x) and the same windows moved one character along (the targets, y).
"""

import math

import numpy as np


def get_batch(data, block_size, batch_size, rng):
    """Return (x, y), two arrays of shape (batch_size, block_size).

    Pick batch_size random start positions with rng (a numpy Generator). Row i of x is
    data[start : start + block_size]; row i of y is the same window moved one place on.
    """
    raise NotImplementedError("Step 2: choose starts with rng.integers, then slice x and y")


def starting_loss(vocab_size):
    """The loss of an untrained model that guesses every character equally: log(vocab_size)."""
    raise NotImplementedError("Step 2: return math.log(vocab_size)")
