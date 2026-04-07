# Contributing And Project Policy

## Mainline Repository Policy

This GitHub repository is the single mainline project for all future work.

That means:

- all training runs are launched from this repository
- all evaluation runs are launched from this repository
- all visualization and example export scripts live in this repository
- all bug fixes are applied in this repository
- all method iterations are implemented in this repository
- all stable config changes are committed in this repository
- all paper-facing figure export paths and helper scripts are organized from this repository

## Working Rule

Do not treat local one-off scripts as the source of truth.

If a script, config, or evaluation path is worth keeping, it must be integrated here, validated here, committed here, and pushed back to GitHub.

## Allowed Local-Only Artifacts

The repository intentionally keeps large or machine-specific artifacts out of git, including:

- `outputs/`
- `data/`
- `.venv/`
- temporary logs
- scratch notes

Representative example figures, lightweight summaries, and export scripts may be added to the repository when they are useful for public release or paper support.

## Preferred Workflow

1. Modify code in this repository.
2. Validate the affected training / eval / example path locally.
3. Commit the change in this repository.
4. Push the change to GitHub.
5. Continue iteration from the updated repository state.
