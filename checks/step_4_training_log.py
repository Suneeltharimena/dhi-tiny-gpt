"""Read the training log"""

import importlib


def _log(fixtures):
    return (fixtures / "practice_log.txt").read_text(encoding="utf-8")


def test_read_log_keeps_only_step_lines(fixtures):
    """read_log returns one (step, train, val) tuple for each evaluation line and ignores the rest"""
    t = importlib.import_module("trainlog")
    rows = t.read_log(_log(fixtures))
    assert len(rows) == 6, f"The practice log has six 'step ...' lines; got {len(rows)} rows. Skip 'iter' lines, checkpoint lines and comments."
    assert tuple(rows[1]) == (250, 1.9802, 2.0611), f"The second row should be (250, 1.9802, 2.0611); got {tuple(rows[1])}. Step is a whole number; the losses are decimals."


def test_best_step_is_the_lowest_validation_loss(fixtures):
    """best_step finds the step where validation loss is lowest, not training loss"""
    t = importlib.import_module("trainlog")
    rows = t.read_log(_log(fixtures))
    assert t.best_step(rows) == 750, f"Validation loss is lowest at step 750 and rises after it, while training loss keeps falling: the model has started to overfit. Got {t.best_step(rows)}."


def test_loss_table_is_markdown(fixtures):
    """loss_table writes a Markdown table with a header, a divider and one line per row"""
    t = importlib.import_module("trainlog")
    lines = t.loss_table([(0, 4.1795, 4.1803), (250, 1.9802, 2.0611)]).splitlines()
    assert len(lines) == 4, f"Two rows need four lines: header, divider and two rows; got {len(lines)}."
    assert [c.strip() for c in lines[0].strip("|").split("|")] == ["Step", "Train loss", "Val loss"], f"The header should read | Step | Train loss | Val loss |; got {lines[0]!r}."
    assert set(lines[1].replace("|", "").replace("-", "").strip()) == set(), f"The second line should be a divider such as |---|---|---|; got {lines[1]!r}."
    assert [c.strip() for c in lines[3].strip("|").split("|")] == ["250", "1.9802", "2.0611"], f"The last row should hold 250, 1.9802 and 2.0611; got {lines[3]!r}. Show losses with four decimal places."


def test_empty_log_gives_no_rows():
    """A log with no evaluation lines gives an empty list instead of an error"""
    t = importlib.import_module("trainlog")
    assert t.read_log("iter 0: loss 4.1720, time 950.00ms, mfu -100.00%\n") == [], "A log with no 'step' line should give an empty list."
