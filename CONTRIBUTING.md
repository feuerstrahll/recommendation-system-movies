# Contributing

## Branches

`main` and `feature/dev` were merged (see git history) and now share the
same project layout — there is no longer a split where models live on one
branch and the cold-start questionnaire on another. Treat `main` as the
default base for new work unless a maintainer asks for a different branch.

## Workflow

1. Branch from `main`.
2. Keep each branch/PR focused on one change.
3. If you copy code from elsewhere (another project, a notebook, a
   generated snippet), preserve attribution and document the destination
   path plus any schema or dependency changes it requires.

Example:

```bash
git switch main
git switch -c docs/short-description
```

## Documentation standards

- Describe what the code currently does, not what a future system might do.
- Mark prototypes, examples, pseudocode, and planned integrations explicitly.
- Do not label a component as production-ready without deployment,
  reliability, security, and load-test evidence.
- Report measured values with the dataset, split, seed, environment, and
  command used to obtain them.
- Do not present illustrative output as a guaranteed result.
- Do not add a license or change licensing terms without repository-owner
  approval.

## Pull requests

Keep each pull request focused. Explain:

- what changed and why;
- how the change was checked (tests run, `evaluation/compare_models.py`
  output, manual verification, etc.);
- any remaining integration or reproducibility limitations.
