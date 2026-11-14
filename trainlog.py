"""Step 4: read the training log.

nanoGPT prints a line like "step 250: train loss 1.9802, val loss 2.0611" at every
evaluation, among many other lines. Turn the log into a loss table and find the best step.
"""


def read_log(text):
    """Return a list of (step, train_loss, val_loss) tuples, one for each "step ..." line, in order.

    Ignore every other line ("iter ...", "saving checkpoint ...", comments). step is an int; the
    losses are floats.
    """
    raise NotImplementedError("Step 4: keep lines that start with 'step ', then split out the three numbers")


def best_step(rows):
    """The step with the lowest validation loss. Training past it is overfitting; nanoGPT keeps this checkpoint."""
    raise NotImplementedError("Step 4: use min() with the validation loss as the key")


def loss_table(rows):
    """Return a Markdown table (a string) for your README: header line, divider line, one line per row.

    Columns: Step | Train loss | Val loss. Show losses to 4 decimal places.
    """
    raise NotImplementedError("Step 4: build '| Step | Train loss | Val loss |', '|---|---|---|', then the rows")
