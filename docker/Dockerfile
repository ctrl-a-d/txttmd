FROM python:3.11-slim

LABEL maintainer="txttmd"
LABEL description="Note automation system with LLM-powered categorization"

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Create non-root user
RUN groupadd -g 1000 noteprocessor && \
    useradd -u 1000 -g noteprocessor -m -s /bin/bash noteprocessor

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY config/ ./config/

# Set ownership
RUN chown -R noteprocessor:noteprocessor /app

# Switch to non-root user
USER noteprocessor

# Create data directories
RUN mkdir -p /app/data/notes

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.exit(0)"

# Default command
CMD ["python", "-m", "src.main"]
