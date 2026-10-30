FROM python:3.11-slim

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
  && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md LICENSE ./
COPY sightline ./sightline
COPY config ./config
COPY app ./app

RUN pip install --no-cache-dir -U pip && pip install --no-cache-dir -e .

EXPOSE 8000
CMD ["uvicorn", "sightline.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
