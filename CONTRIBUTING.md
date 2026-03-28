# Contributing to AgentRunKit

Thanks for your interest in contributing.

## Setup

```bash
make bootstrap
make check
```

`make bootstrap` installs [Mint](https://github.com/yonaskolb/Mint) and the pinned versions of SwiftFormat and SwiftLint from the `Mintfile`. `make check` runs formatting, linting, and tests.

## Documentation

New public API must include `///` doc comments (one sentence, period). If the feature adds a new concept or workflow, add or update an article in `Sources/AgentRunKit/Documentation.docc/Articles/`. Run `make docs` to verify the DocC build produces zero warnings.

## Before Submitting a PR

Run `make check`. If it passes locally, CI will pass.
