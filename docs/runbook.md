# Local operator runbook

Run from the project directory with the virtual environment active. Put `--root` before the command. Examples assume a new registry and training versions 1 and 2.

## Evaluate and release

~~~sh
ROOT=.local/manual
taxi --root "$ROOT" ingest
# Set SNAP to the printed snapshot ID.
taxi --root "$ROOT" train "$SNAP"
taxi --root "$ROOT" bootstrap 1 --reason "Median passed the serving check"
taxi --root "$ROOT" evaluate "$SNAP" --candidate 2 --baseline 1
# Set DECISION to the printed ID. Review checks and slice counts.
taxi --root "$ROOT" promote "$DECISION" --reason "Reviewed February results"
taxi --root "$ROOT" test-report "$DECISION"
~~~

A failed validation decision stays in `decisions/`. If the approved alias moved after evaluation, re-evaluate against the current incumbent. Review completed releases under `release/history/`. Keep March for held-out reporting and use February for model selection. For the rejection drill, run `train "$SNAP" --weak`, evaluate the versions it returns, and attempt to promote that decision. The command must fail.

## Recover a scoring partition

~~~sh
taxi --root "$ROOT" score "$SNAP" 2024-04-01 --fail after_stage
taxi --root "$ROOT" score "$SNAP" 2024-04-01
taxi --root "$ROOT" score "$SNAP" 2024-04-01
~~~

The injected failure stops before publication. The first retry rebuilds staged files using the pinned version. The next retry returns the same verified manifest. For a failure after commit, use `--fail after_commit` on a fresh date. The retry checks the committed files and publishes the pointer. Inspect `predictions/<date>/plans/`, `runs/`, and `active.json`. A partial run requires the original source revision. If the active run changed while a replacement was pending, review the newer run before starting a new replacement ID.

~~~sh
taxi --root "$ROOT" rollback 1 --reason "Restore the previous approved version"
taxi --root "$ROOT" backfill "$SNAP" 2024-04-01 2024-04-01 --version 1 --run-id rollback-replay
~~~

Use the rollback commands after a successful promotion. To exercise the full sequence when the TLC candidate fails its gate, run `python scripts/replay.py --root .local/rollback-demo --fixture` in a fresh directory. It runs the same release and recovery commands on synthetic trips.

The rollback changes future model resolution. Backfill supersedes the selected partition and retains both immutable outputs. For an interrupted range, retry the same dates with the same run ID. Each date commits separately.

## Monitor as outcomes arrive

~~~sh
taxi --root "$ROOT" monitor 2024-04-01 --as-of 2024-04-01T23:59:59-04:00 --decision "$DECISION"
taxi --root "$ROOT" monitor 2024-04-01 --as-of 2024-04-04T06:00:00-04:00 --decision "$DECISION"
~~~

Run both before rollback to compare the same predictions. Input shift uses total variation distance against February for pickup zone and hour, with a 0.10 threshold. The error flag needs the outcome window to close, at least 1,000 joined labels, and MAE above 115% of the same version's validation reference. Inspect label coverage and signed error with the result. These cutoffs are demo settings.
