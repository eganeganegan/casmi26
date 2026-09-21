# GitHub Copilot guidance

The repository-wide engineering and research rules are defined in `AGENTS.md` and
`CONTRIBUTING.md`. GitHub Copilot's automatically discovered instructions live in
`.github/copilot-instructions.md` and intentionally summarize the same constraints.

Suggested code must preserve structure-disjoint validation, deterministic behavior, bounded-memory
data processing, the ignored artifact boundary, and the offline Kaggle submission contract. Generated
code is not complete until its tests and documentation are updated.
