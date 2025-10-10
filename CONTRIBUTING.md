## Contributing

Thank you for your interest in contributing!

### Dev setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install pre-commit
pre-commit install
```

### Testing & linting
```bash
ruff check .
pytest -q
```

### Pull requests
- Write clear descriptions and include repro steps.
- Add/adjust tests for new behavior when reasonable.
- Keep public APIs documented.


