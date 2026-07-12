FROM --platform=linux/amd64 python:3.11-slim

# ffmpeg + ffprobe for frame extraction. No pip deps: agent.py uses stdlib only,
# which keeps the image small and the build fast.
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY agent.py .

# Your key is baked in at build time (Track 2 injects nothing).
# Pass it with:  docker buildx build --build-arg GEMINI_API_KEY=xxx ...
# NOTE: the image is public. Use a throwaway key and revoke it after judging.
ARG GEMINI_API_KEY=""
ENV GEMINI_API_KEY=${GEMINI_API_KEY}
ENV GEMINI_MODEL="gemini-3.1-flash-lite"
ENV PYTHONUNBUFFERED=1

CMD ["python", "agent.py"]
