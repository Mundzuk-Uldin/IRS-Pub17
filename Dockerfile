# pub17 API on CPU. The dev machine runs the models on a GPU; in this image
# they run on CPU, which is slower but needs no GPU runtime to review.
FROM python:3.13-slim

WORKDIR /app

# CPU-only torch first, so sentence-transformers doesn't pull the CUDA build.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake both models into the image so a container starts without downloading.
ENV HF_HOME=/models
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('BAAI/bge-base-en-v1.5'); CrossEncoder('BAAI/bge-reranker-base')"

COPY pub17/ pub17/

EXPOSE 8000
CMD ["uvicorn", "pub17.api:app", "--host", "0.0.0.0", "--port", "8000"]
