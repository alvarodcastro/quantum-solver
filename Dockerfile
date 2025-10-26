# Base image: Python + Node
FROM python:3.8-slim AS base

# Install OS-level dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl vim \
    && rm -rf /var/lib/apt/lists/*

# Install Node (if not already present) — here we use Node 19 from NodeSource
RUN curl -fsSL https://deb.nodesource.com/setup_19.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy Python dependency files
COPY pyproject.toml setup.cfg /app/

# Copy source code
COPY src/ /app/src/
COPY web/src/flask-server/ /app/web/src/flask-server/
COPY web/ /app/web/

# Install Python dependencies
RUN pip install --no-cache-dir -e .

# Install frontend (Node) dependencies
WORKDIR /app/web
RUN npm install

# Go back to root workdir
WORKDIR /app

# Expose ports
EXPOSE 5000
EXPOSE 3000

# Set environment variables (you can supply IBMQ_TOKEN at runtime)
ENV PYTHONUNBUFFERED=1

# Entrypoint: run both backend and frontend
# Use a shell command to start both processes:
CMD ["bash", "-c", "\
     cd /app/web/src/flask-server && python server.py & \
     cd /app/web && npm run dev \
"]
