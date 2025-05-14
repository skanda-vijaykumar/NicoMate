#!/bin/bash
set -e

# Wait for Ollama to be ready
echo "Waiting for Ollama to be ready..."
for i in {1..30}; do
  if curl -s http://ollama:11434/api/tags >/dev/null; then
    echo "Ollama is ready!"
    break
  fi
  echo "Waiting for Ollama... ($i/30)"
  sleep 2
done

if [ $i -eq 30 ]; then
  echo "Ollama did not become ready in time"
  exit 1
fi

# List of models to pull
models=("llama3.1" "nomic-embed-text")

# Pull each model
for model in "${models[@]}"; do
  echo "Pulling model: $model"
  curl -X POST http://ollama:11434/api/pull -d "{\"name\":\"$model\"}"
  if [ $? -eq 0 ]; then
    echo "Successfully pulled $model"
  else
    echo "Failed to pull $model"
  fi
done

echo "All models initialized!"