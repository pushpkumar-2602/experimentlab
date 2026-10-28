FROM python:3.11-slim

WORKDIR /app

# Install dependencies first, in their own layer (see the note in Step 60).
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Then copy our code.
COPY . .

# Hosting platforms (like Render) tell us which port to use through the PORT
# environment variable. If it isn't set (local Docker), we use 8000.
# 0.0.0.0 means "accept connections from outside the container".
CMD ["sh", "-c", "python -m uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
