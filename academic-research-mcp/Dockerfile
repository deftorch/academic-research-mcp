# Stage 1: Builder
FROM python:3.12-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install dependencies into a virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Runtime
FROM python:3.12-slim

WORKDIR /app

# Install runtime dependencies (if any system libs are needed)
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     libgomp1 \
#     && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV ACADEMIC_CONTACT_EMAIL="your.email@university.edu"

# Copy application code
COPY . .

# Create cache directory
RUN mkdir -p .cache_data && chmod 777 .cache_data
RUN mkdir -p paper_vectordb && chmod 777 paper_vectordb

# Command to run the server
CMD ["python", "academic_research_server.py"]
