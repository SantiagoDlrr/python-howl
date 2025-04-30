# Run tests in container
docker-compose run fastapi-backend pytest

# Run with coverage
docker-compose run fastapi-backend pytest tests/ --cov=main --cov=temp_storage --cov-report=term

# Run specific test file
docker-compose run fastapi-backend pytest tests/test_main.py

# Run with html coverage for a gui
docker-compose run fastapi-backend pytest tests/ --cov=mainbut e --cov-report=html