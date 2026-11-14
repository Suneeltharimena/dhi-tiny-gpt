"""Step 3: train a tiny model on CPU with numpy.

nanoGPT needs a GPU, so here you train the simplest next-character model: a bigram. It
holds one table of scores (logits). Row c is the model's scores for the character that
follows c. Turning scores into probabilities (softmax) and measuring the miss
(cross-entropy) are the same two ideas a GPT uses.
"""

import numpy as np

from batches import get_batch


def loss(logits, data):
    """The average cross-entropy of the model on data (a list or array of ids): lower is better.

    For each pair of neighbours (data[i], data[i + 1]): softmax row data[i] of logits, then take
    minus the log of the probability given to data[i + 1]. Return the average over all pairs.
    """
    raise NotImplementedError("Step 3: softmax each row, then average -log(probability of the true next character)")


def train(train_data, vocab_size, steps=300, lr=1.0, batch_size=32, block_size=8, seed=0):
    """Train by gradient descent and return the logits table, shape (vocab_size, vocab_size).

    Start from zeros (the model guesses evenly, so its loss is log(vocab_size)). Each step: take a
    batch with get_batch(train_data, block_size, batch_size, rng), then move every score a small
    step downhill. For one batch, the gradient of the average loss is (probabilities - one-hot
    target) / number of pairs, added up in the row of each input character (np.add.at helps).
    Create rng once with np.random.default_rng(seed) so every run gives the same answer.
    """
    raise NotImplementedError("Step 3: loop over steps, take a batch, work out the gradient, subtract lr times it")
