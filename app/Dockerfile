# Base image with Python 3.10
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install dependencies and ffmpeg (required by whisper)
# RUN apt-get update && apt-get install -y ffmpeg && \
#     apt-get clean && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy requirements file (if using pip freeze) OR install manually
# COPY requirements.txt .
COPY requirements2.txt .

# Install Python dependencies
# RUN pip install  -r requirements.txt
RUN pip install --no-cache-dir -r requirements2.txt

COPY requirements3.txt .
RUN pip install --no-cache-dir -r requirements3.txt

COPY requirements4.txt .
RUN pip install --no-cache-dir -r requirements4.txt


# Copy app files into the container
COPY . .

#Uncomment below to use https
# COPY cert.pem /app/cert.pem
# COPY key.pem /app/key.pem
# EXPOSE 443

# Uncomment below for local development
EXPOSE 8000 


#Uncomment below to use https
# CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "443", "--ssl-keyfile", "key.pem", "--ssl-certfile", "cert.pem"]

# Uncomment below for local development
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]