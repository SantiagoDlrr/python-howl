# Howl Python Tests

This directory contains unit tests for the Howl Python application.

## Running Tests

To run all tests:

```bash
pytest
```

To run tests with verbose output:

```bash
pytest -v
```

To run a specific test file:

```bash
pytest tests/test_main.py
```

To run a specific test class:

```bash
pytest tests/test_main.py::TestEndpoints
```

To run a specific test method:

```bash
pytest tests/test_main.py::TestEndpoints::test_root_endpoint
```

## Test Coverage

To run tests with coverage report:

```bash
pytest --cov=python_howl
```

## Test Structure

- `test_main.py`: Tests for the main FastAPI application
- `test_temp_storage.py`: Tests for the transcript storage functionality

## Mocking

The tests use pytest's monkeypatch and unittest.mock to mock external dependencies like:

- Gemini API
- OCI Language client
- File operations
- Environment variables

This allows testing the code without actual API calls or file system operations.
