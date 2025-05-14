FROM python:3.10-slim

WORKDIR /app

# Install build dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create necessary directories
RUN mkdir -p static templates data extracted_best

# Copy application code
COPY app/ ./app/
COPY setup_dirs.sh run.py ./

# Make the script executable
RUN chmod +x setup_dirs.sh

# Make sure directories are set up properly
RUN ./setup_dirs.sh

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
CMD ["python", "run.py"]