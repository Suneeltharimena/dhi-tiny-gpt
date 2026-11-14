"""Check your project, step by step.

    python grade.py --all        check every step
    python grade.py --step 2     check one step
    python grade.py --live       one run with your own API key: prints score and cost, never pass or fail

The checks never call a model. Where your code would call one, they replay saved replies from
fixtures/, so every run is free and gives the same result. Your GitHub repository runs the same
command on every push (.github/workflows/grade.yml): a green tick means every step passes.
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CHECKS = HERE / "checks"


def steps() -> list[tuple[int, Path, str]]:
    """Each checks/step_N_*.py file is one build step; its first docstring line is the step's name."""
    found = []
    for path in sorted(CHECKS.glob("step_*.py")):
        number = int(path.stem.split("_")[1])
        doc = ast.get_docstring(ast.parse(path.read_text(encoding="utf-8"))) or path.stem
        found.append((number, path, doc.splitlines()[0]))
    return sorted(found)


class Report:
    """A pytest plugin that collects each check's result and prints it in plain English."""

    def __init__(self) -> None:
        self.results: list[tuple[str, str, bool, str]] = []  # (step file, check name, passed, message)

    def pytest_runtest_logreport(self, report) -> None:  # noqa: ANN001 (pytest hook)
        # The call is the check itself; a set-up or tear-down that fails, or a skipped check, also counts as not passed.
        if report.when == "call" or report.failed or report.skipped:
            name = report.nodeid.split("::")[-1]
            message = ""
            if report.failed:
                crash = getattr(report.longrepr, "reprcrash", None)
                message = crash.message if crash is not None else str(report.longrepr).strip().splitlines()[-1]
            elif report.skipped:
                message = "This check was skipped, so it cannot pass."
            self.results.append((report.nodeid.split("::")[0], name, report.passed and report.when == "call", message))


def describe(step_file: Path, name: str) -> str:
    """The check's docstring, or its name made readable."""
    tree = ast.parse(step_file.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            doc = ast.get_docstring(node)
            if doc:
                return doc.splitlines()[0]
    return name.removeprefix("test_").replace("_", " ").capitalize()


def plain(message: str) -> str:
    """Strip pytest's 'AssertionError: ' prefix and its expression dump; keep the sentence written for you."""
    text = message.split("\nassert ", 1)[0]
    for prefix in ("AssertionError: ", "assert "):
        text = text.removeprefix(prefix)
    return text.strip()


def run(selected: list[tuple[int, Path, str]], src: Path) -> bool:
    import pytest

    os.environ["GRADE_SRC"] = str(src)
    report = Report()
    args = [str(path) for _, path, _ in selected]
    args += ["-q", "-p", "no:cacheprovider", "--rootdir", str(HERE), "-c", str(CHECKS / "pytest.ini"), "--no-header", "-rN", "--tb=line"]
    # pytest's own output is hidden; this script prints the results in plain English below.
    with open(os.devnull, "w", encoding="utf-8") as quiet:
        stdout = sys.stdout
        sys.stdout = quiet
        try:
            pytest.main(args, plugins=[report])
        finally:
            sys.stdout = stdout

    passed_steps = 0
    for number, path, title in selected:
        rows = [row for row in report.results if Path(row[0]).name == path.name]
        ok = bool(rows) and all(row[2] for row in rows)
        passed_steps += ok
        print(f"\nStep {number}: {title}  [{'PASS' if ok else 'NOT YET'}]")
        if not rows:
            print("  The checks for this step could not run. Look for an error near the top of your file.")
        for _, name, passed, message in rows:
            print(f"  {'pass' if passed else 'FAIL'}  {describe(path, name)}")
            if not passed and message:
                print(f"        What to look at: {plain(message)}")
    print(f"\n{passed_steps} of {len(selected)} {'step passes' if len(selected) == 1 else 'steps pass'}.")
    return passed_steps == len(selected)


def live(src: Path) -> int:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("A live run needs your own API key in ANTHROPIC_API_KEY.")
        print("Set it as a secret in your codespace (or export it in your terminal), then run: python grade.py --live")
        return 2
    hook = HERE / "live.py"
    if not hook.exists():
        print("This project has no live run: every check already uses your own code and data.")
        return 0
    sys.path.insert(0, str(src))
    spec = importlib.util.spec_from_file_location("live", hook)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    result = module.main()
    print(f"\nLive run on {result['cases']} saved cases with {result['model']}")
    print(f"  Score: {result['score']:.0%}")
    print(f"  Cost of this run: ${result['cost_usd']:.4f} (prices of {result['prices_as_of']})")
    print("A live run is for your results table. It never passes or fails: models change, and the saved checks do not.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check your project, step by step.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--all", action="store_true", help="check every step (the default)")
    group.add_argument("--step", type=int, metavar="N", help="check one step")
    group.add_argument("--live", action="store_true", help="one run with your own API key: score and cost, never pass or fail")
    parser.add_argument("--src", default=".", help=argparse.SUPPRESS)  # where the code under test lives
    args = parser.parse_args(argv)

    src = (HERE / args.src).resolve()
    if args.live:
        return live(src)
    every = steps()
    if args.step is not None:
        chosen = [item for item in every if item[0] == args.step]
        if not chosen:
            print(f"There is no step {args.step}. Steps are {', '.join(str(item[0]) for item in every)}.")
            return 2
    else:
        chosen = every
    return 0 if run(chosen, src) else 1


if __name__ == "__main__":
    sys.exit(main())
