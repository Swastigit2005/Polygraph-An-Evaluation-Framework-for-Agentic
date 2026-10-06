# Slim CPU image for the Gradio demo. Build:  docker build -t agentic-rag-evals .
# Run:  docker run --rm -p 7860:7860 --env-file .env agentic-rag-evals
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.hf \
    HF_HUB_DISABLE_TELEMETRY=1 \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

# Bake the local embedding + reranker models into the image so startup needs no download.
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('BAAI/bge-small-en-v1.5'); CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

COPY src/ src/
COPY data/ data/
COPY eval/ eval/
COPY app.py .

RUN useradd --create-home --uid 1000 app && chown -R app /app
USER app

EXPOSE 7860
CMD ["python", "app.py"]
