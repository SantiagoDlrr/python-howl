# Base image with Python 3.10
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install dependencies and ffmpeg (required by whisper)
RUN apt-get update && apt-get install -y ffmpeg && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy requirements file (if using pip freeze) OR install manually
COPY requirements.txt .
COPY requirements2.txt .

# Install Python dependencies
RUN pip install  -r requirements.txt
RUN pip install  -r requirements2.txt

# COPY requirements3.txt .
# RUN pip install --no-cache-dir -r requirements3.txt

# Copy app files into the container
COPY . .

# Expose port (FastAPI default)
EXPOSE 8000

# Run the FastAPI app
CMD ["uvicorn", "main:app","--reload", "--host", "0.0.0.0", "--port", "8000"]
