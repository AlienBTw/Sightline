FROM python:3.11-slim

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY sightline ./sightline
COPY app ./app
COPY .streamlit ./.streamlit

RUN pip install --no-cache-dir -U pip && pip install --no-cache-dir -e .

EXPOSE 8501
ENV API_URL=http://api:8000
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.address", "0.0.0.0", "--server.port", "8501"]
