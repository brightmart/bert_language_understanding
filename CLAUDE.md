# Project Setup

This file contains project-specific information for Claude.

## Testing Commands
- Run tests: poetry run test or poetry run tests
- Run with coverage: poetry run pytest --cov
- Run unit tests only: poetry run pytest tests/unit
- Run integration tests only: poetry run pytest tests/integration
- Run with verbose output: poetry run pytest -v
- Run specific test file: poetry run pytest tests/test_file.py
- Run tests with specific marker: poetry run pytest -m unit

Note: Coverage threshold is currently set to 0% for initial setup. Update to 80% in pyproject.toml when ready.

## Code Quality Commands
- Linting: TBD
- Type checking: TBD

## Project Structure
- Main source code: model/ and root .py files
- Tests: tests/
- Documentation: docs/
- Testing config: pyproject.toml

