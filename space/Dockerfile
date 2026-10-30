# Image for the Hugging Face Space (Docker SDK). Build locally from the project root:
#     docker build -f space/Dockerfile -t freespace .
# space/deploy.py copies this file to the Space's root and every other file to the
# same relative path it has here, so the COPY lines work in both places.
FROM python:3.12-slim
# links the ghcr.io package to the GitHub repo
LABEL org.opencontainers.image.source=https://github.com/BhaavinJain/Monocular-Free-Space-Perception

# HF Spaces run the container as uid 1000; own everything as that user, never root
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    MODEL_DIR=/home/user/app/models \
    SAMPLE_DIR=/home/user/app/models/samples \
    ORT_THREADS=2
WORKDIR /home/user/app

COPY --chown=user space/requirements.txt space/requirements.txt
RUN pip install --no-cache-dir --user -r space/requirements.txt

# Models and gallery at BUILD time, pinned to one commit of the public model repo:
# a cold start never downloads anything, and changing the served model is a
# one-line, reviewable diff of MODEL_REVISION. (ARGs are env vars inside RUN.)
# The download's bookkeeping (~/.cache/huggingface: telemetry list + xet log;
# models/.cache: lock and metadata files) is removed IN THE SAME STEP: a later
# RUN rm would hide the files but leave them in this layer of the public image.
ARG MODEL_REPO=BhaavinJain/monocular-freespace-models
ARG MODEL_REVISION=d272633e7b8f04109ead2adcb2b97d99299e423e
RUN python -c "import os; from huggingface_hub import snapshot_download; snapshot_download(os.environ['MODEL_REPO'], revision=os.environ['MODEL_REVISION'], local_dir='models')" \
 && rm -rf /home/user/.cache/huggingface models/.cache

COPY --chown=user pipeline.py bev.py camera.py ground_plane.py planner.py pointcloud.py scale.py metrics.py ./
COPY --chown=user backend/app.py backend/app.py

EXPOSE 7860
CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "7860"]
