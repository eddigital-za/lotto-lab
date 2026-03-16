FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONUNBUFFERED=1

EXPOSE ${PORT:-5000}

CMD gunicorn --bind 0.0.0.0:${PORT:-5000} --workers 2 --timeout 120 --preload wsgi:application
