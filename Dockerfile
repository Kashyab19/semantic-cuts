# Use a slim Python image (lightweight and fast)
# If you need GPU support later, change this to an nvidia/cuda image or pytorch/pytorch
FROM python:3.10-slim

# Set the working directory to match the volume mount in your compose file
WORKDIR /app

# Install system dependencies (often needed for AI/Inference libraries like OpenCV or numpy)
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the code
COPY . .

# This CMD is a fallback; your docker-compose 'command' overrides this.
CMD ["python3", "app/server.py"]
