# Local Ollama server for Muse Glimmer 30B (report writer, labeling, optional VLM gate).
# Pull the model once after start: docker compose exec ollama ollama pull muse-glimmer:30b
# GPU: add a device reservation for your GPU in docker-compose.yml, or run Ollama natively on the host.
FROM ollama/ollama:latest
EXPOSE 11434
