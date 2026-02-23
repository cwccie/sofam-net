FROM python:3.11-slim

LABEL maintainer="Corey A. Wade <corey@coreywade.com>"
LABEL description="SOFAM-Net: Federated Learning for Cross-Organization Threat Detection"

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY pyproject.toml README.md ./
COPY src/ src/
RUN pip install --no-cache-dir -e ".[dev]"

# Default port for coordinator API
EXPOSE 8080
# Dashboard port
EXPOSE 5000

# Default: start coordinator
CMD ["sofam", "start-coordinator", "--host", "0.0.0.0", "--port", "8080"]
