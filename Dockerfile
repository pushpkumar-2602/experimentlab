FROM python:3.11-slim

WORKDIR /app

# Install dependencies first, in their own layer (see the note below).
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Then copy our code.
COPY . .

# 0.0.0.0 means "accept connections from outside the container".
CMD ["python", "-m", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
