# This is the Dockerfile for the FastAPI app.
# location: Dockerfile

FROM python:3.13.2-slim

# Set working directory
WORKDIR /app

# Set non-sensitive environment variables
ARG APP_ENV=production
ARG POSTGRES_URL
ARG JWT_SECRET_KEY
ARG LLM_API_KEY
ARG OPENAI_API_KEY

ENV APP_ENV=${APP_ENV} \
    PYTHONFAULTHANDLER=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=random \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    POSTGRES_URL=${POSTGRES_URL} \
    JWT_SECRET_KEY=${JWT_SECRET_KEY} \
    LLM_API_KEY=${LLM_API_KEY} \
    OPENAI_API_KEY=${OPENAI_API_KEY}

# Install system dependencies (and bash)
RUN apt-get update && apt-get install -y \
    bash \
    build-essential \
    libpq-dev \
  && pip install --upgrade pip \
  && pip install uv \
  && rm -rf /var/lib/apt/lists/*

# Copy pyproject.toml first to leverage Docker cache
COPY pyproject.toml .
RUN uv venv && . .venv/bin/activate && uv pip install -e .

# Explicitly install langchain-ollama to ensure it's available
RUN . .venv/bin/activate && pip install langchain-ollama

# Copy the application
COPY . .

# Install dos2unix, convert all scripts to LF, and ensure they're executable
RUN apt-get update && apt-get install -y dos2unix \
 && find /app/scripts -type f -name '*.sh' -exec dos2unix {} \; \
 && find /app/scripts -type f -name '*.sh' -exec chmod +x {} \; \
 && rm -rf /var/lib/apt/lists/*

# Make entrypoint script executable - do this before changing user
RUN chmod +x /app/scripts/docker-entrypoint.sh

# Create a non-root user
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

# Create log directory
RUN mkdir -p /app/logs

# Default port
EXPOSE 8000

# Log the environment we're using
RUN echo "Using ${APP_ENV} environment"

# Command to run the application
ENTRYPOINT ["/app/scripts/docker-entrypoint.sh"]
CMD ["/app/.venv/bin/uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"] 