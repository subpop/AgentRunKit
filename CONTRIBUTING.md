# Contributing to AgentRunKit

Thanks for your interest in contributing.

## Setup

```bash
make bootstrap
make check
```

`make bootstrap` installs [Mint](https://github.com/yonaskolb/Mint) and the pinned versions of SwiftFormat and SwiftLint from the `Mintfile`. `make check` runs formatting, linting, and tests.

## Before Submitting a PR

Run `make check`. If it passes locally, CI will pass.
