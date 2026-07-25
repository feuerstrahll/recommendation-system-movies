# Contributing

## Source ownership

Use the repository area that owns the component:

| Change | Base source |
|---|---|
| Recommendation model or model experiment | `main/models` |
| Cold-start questionnaire | `feature/dev/recommendation-system-lab5/models/cold_start_questionnaire.py` |

Do not use the whole `feature/dev` branch as the default model source solely
because the questionnaire lives there.

## Workflow

1. Start model changes from `main`, unless a maintainer requests a different
   base branch.
2. Create a focused working branch.
3. Keep changes to the questionnaire separate from unrelated experimental files
   in `feature/dev`.
4. If code is copied between branches, preserve attribution and document the
   destination path and any schema or dependency changes.
5. Avoid merging an entire branch only to obtain one component.

Example:

```bash
git switch main
git switch -c docs/short-description
```

## Documentation standards

- Describe what the code currently does, not what a future system might do.
- Mark prototypes, examples, pseudocode, and planned integrations explicitly.
- Do not label a component as production-ready without deployment, reliability,
  security, and load-test evidence.
- Report measured values with the dataset, split, seed, environment, and command
  used to obtain them.
- Do not present illustrative output as a guaranteed result.
- Do not add a license or change licensing terms without repository-owner
  approval.

## Pull requests

Keep each pull request focused. Explain:

- the source branch and file;
- what changed and why;
- how the change was checked;
- any remaining integration or reproducibility limitations.

When a change combines `main/models` with the cold-start questionnaire from
`feature/dev`, include an integration test or a small reproducible example.
