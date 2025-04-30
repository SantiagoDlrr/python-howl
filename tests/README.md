# Run tests in container
docker-compose run fastapi-backend pytest

# Run with coverage
docker-compose run fastapi-backend pytest tests/ --cov=app

# Run specific test file
docker-compose run fastapi-backend pytest tests/test_main.py