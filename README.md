# Train a tiny GPT

A Dhi graded project (AI engineering, How models learn). Brewly is a practice company: the training text and the training log in `fixtures/` are invented practice data.

The real project trains nanoGPT on a book you choose, on a free GPU notebook. This kit grades the code parts you can check on a laptop with no GPU: the tokeniser, the batches and starting loss, a tiny training loop, and the log reader for your loss table.

## Start

1. Click **Use this template** on GitHub to make your own public repository, then open it in a codespace.
2. `pip install -r requirements.txt`
3. `python grade.py --all`. Every step says NOT YET until you build it.
4. Add two lines to your `.gitignore` for step 5: `ckpt.pt` and `out-*/`.

Python 3.12 or later. `grade.py --all` takes under 10 seconds on a laptop (about 1 second measured on the model answer).

## The steps

| Step | You build | Check it |
|---|---|---|
| 1 | `build_vocab()`, `encode()`, `decode()` and `split_data()` in `tokeniser.py` | `python grade.py --step 1` |
| 2 | `get_batch()` and `starting_loss()` in `batches.py` | `python grade.py --step 2` |
| 3 | `loss()` and `train()` in `bigram.py` (needs step 1) | `python grade.py --step 3` |
| 4 | `read_log()`, `best_step()` and `loss_table()` in `trainlog.py` | `python grade.py --step 4` |
| 5 | `ckpt.pt` and `out-*/` in your `.gitignore`; no key or checkpoint file anywhere | `python grade.py --step 5` |

Push to GitHub and the **Actions** tab runs `python grade.py --all`. A green tick means every step passes.

## How the checks work

The checks never call a model and never open the network. Step 3 trains a bigram model (one table of scores for the next character) on `fixtures/brewly_practice.txt` with a fixed seed, so every run gives the same result. Step 4 reads `fixtures/practice_log.txt`, written in nanoGPT's log format with invented numbers. When you run nanoGPT for real, point `read_log()` at your own log and put `loss_table()` in your README.

## What the checks do not prove

They prove your tokeniser, batches, training loop and log reader on practice data. They do not prove your nanoGPT run on a GPU, your loss table, your samples, your controlled experiment or your 500-word write-up. Mark those yourself with the guide in the Dhi lesson.

## Licence

MIT (see `LICENSE`). The code you add is yours.
