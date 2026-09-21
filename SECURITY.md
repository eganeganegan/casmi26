# Security policy

## Reporting a vulnerability

Do not open a public issue for leaked credentials or a vulnerability that could expose private data.
Contact the repository owner privately through GitHub and include the affected file or component,
impact, and reproduction steps. Rotate any exposed token immediately; deleting it from the latest
commit is not sufficient because Git retains history.

## Supported version

Security fixes target the current `main` branch. This research repository does not provide support
for historical experiment snapshots.

## Sensitive material

Kaggle credentials, competition data, external licensed databases, model artifacts, and submission
files must remain outside Git. See `CONTRIBUTING.md` and `.gitignore` for the repository boundary.
