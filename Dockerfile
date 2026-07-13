FROM --platform=linux/amd64 python:3.11-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY agent.py .

ARG GEMINI_API_KEY=""
ENV GEMINI_API_KEY=${GEMINI_API_KEY}
ENV GEMINI_MODEL="gemini-3.1-flash-lite"
ENV PYTHONUNBUFFERED=1

CMD ["python", "agent.py"]
