# Use Python 3.12 to match your project requirements
FROM python:3.12-slim

# 1. Install System Dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 2. Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# 3. Set up the application
WORKDIR /app

# 4. Install Dependencies
# Only copy pyproject.toml (since we deleted uv.lock)
COPY pyproject.toml ./

# Run sync WITHOUT --frozen (this resolves dependencies fresh)
RUN uv sync

# CRITICAL: Add the virtual environment to the PATH
ENV PATH="/app/.venv/bin:$PATH"

# 5. Copy the Code
COPY . .

# 6. Default Command
CMD ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8000"]
