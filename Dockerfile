FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# System dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# The fine-tuned LoRA adapters directory.
# Either bake it into the image (COPY above handles that if present)
# or mount it at runtime:  -v /path/to/final-resume-model:/app/final-resume-model
ENV RESUME_MODEL_PATH=/app/final-resume-model

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]