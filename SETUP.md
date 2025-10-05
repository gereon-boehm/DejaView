# Quick Setup Guide

## 1. Customize Your Project

1. **Update `pyproject.toml`**: Change `name`, `description`, and `authors`
2. **Rename the package**: `src/your_package/` → `src/your_actual_package_name/`
3. **Update imports**: Fix package name in `tests/test_main.py`

## 2. Get Started

```bash
# Install UV (if needed)
brew install uv

# Setup project
uv sync --extra dev
git init
uv run pre-commit install --hook-type commit-msg

# Test everything works
uv run pytest && uv run ruff check && uv run mypy src/
```

## 3. Make Your First Commit

```bash
git add .
git commit -m "feat: initial project setup"
```

## 4. Development Commands

- **Run tests**: `uv run pytest`
- **Format code**: `uv run ruff check --fix && uv run ruff format`
- **Type check**: `uv run mypy src/`
- **Add dependencies**: `uv add package-name`

## 5. Commit Message Format

Use [Conventional Commits](https://www.conventionalcommits.org/):

```bash
feat: add new feature
fix: resolve bug
docs: update documentation
test: add tests
chore: update dependencies
```

---

**That's it!** Delete this file when you're done. 🚀
