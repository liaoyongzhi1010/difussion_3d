"""Training, evaluation, and seed-harness entry points.

- `train_loop`: clean v5 training entry (config-driven, 3-arg seeding).
- `eval_loop`: runs DDIM sampling + the metric suite on a checkpoint.
- `seed_harness`: multi-seed driver with mean+/-95% CI aggregation.

Each module is runnable via `python -m amodal_scene_diff.engine.<name>`.
"""
